import argparse
from collections import deque
from functools import partial
import json
import os
import time

from PIL import Image
import cv2
import numpy as np
import torch
from torchvision.transforms import transforms

from vit_models.model import ViTPose
from vit_utils.top_down_eval import keypoints_from_heatmaps
from vit_utils.visualization import draw_points_and_skeleton, joints_dict

try:  # Add bools -> error stack
    import pycuda.driver as cuda
    import pycuda.autoinit
    import utils_engine as engine_utils
    import tensorrt as trt
    has_trt = True
except ModuleNotFoundError:
    pass

try:
    import onnxruntime
    has_onnx = True
except ModuleNotFoundError:
    pass

__all__ = ['inference']


def pad_image(image: np.ndarray, aspect_ratio: float) -> np.ndarray:
    # Get the current aspect ratio of the image
    image_height, image_width = image.shape[:2]
    current_aspect_ratio = image_width / image_height

    left_pad = 0
    top_pad = 0
    # Determine whether to pad horizontally or vertically
    if current_aspect_ratio < aspect_ratio:
        # Pad horizontally
        target_width = int(aspect_ratio * image_height)
        pad_width = target_width - image_width
        left_pad = pad_width // 2
        right_pad = pad_width - left_pad

        padded_image = np.pad(image, pad_width=((0, 0), (left_pad, right_pad), (0, 0)), mode='constant')
    else:
        # Pad vertically
        target_height = int(image_width / aspect_ratio)
        pad_height = target_height - image_height
        top_pad = pad_height // 2
        bottom_pad = pad_height - top_pad

        padded_image = np.pad(image, pad_width=((top_pad, bottom_pad), (0, 0), (0, 0)), mode='constant')
    return padded_image, (left_pad, top_pad)


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@torch.no_grad()
def inference(img: np.ndarray, target_size: tuple[int, int], model,
              device: torch.device) -> np.ndarray:

    # Prepare input data
    org_h, org_w = img.shape[:2]
    img_tensor = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize((target_size[1], target_size[0])),
                                     ])(img).unsqueeze(0).to(device)

    # Feed to model
    heatmaps = model(img_tensor).detach().cpu().numpy()
    points, prob = keypoints_from_heatmaps(heatmaps=heatmaps,
                                           center=np.array([[org_w // 2,
                                                             org_h // 2]]),
                                           scale=np.array([[org_w, org_h]]),
                                           unbiased=True, use_udp=True)

    points = np.concatenate([points[:, :, ::-1], prob], axis=2)
    return points


def inference_onnx(img: np.ndarray, target_size: tuple[int, int], ort_session,
                   device: torch.device) -> np.ndarray:

    # Prepare input data
    org_h, org_w = img.shape[:2]
    img_input = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    img_input = img_input.astype(np.float32).transpose(2, 0, 1)[None, ...] / 255

    # Feed to model
    ort_inputs = {ort_session.get_inputs()[0].name: img_input}
    heatmaps = ort_session.run(None, ort_inputs)[0]
    points, prob = keypoints_from_heatmaps(heatmaps=heatmaps,
                                           center=np.array([[org_w // 2,
                                                             org_h // 2]]),
                                           scale=np.array([[org_w, org_h]]),
                                           unbiased=True, use_udp=True)

    points = np.concatenate([points[:, :, ::-1], prob], axis=2)
    return points


def inference_trt(img: np.ndarray, target_size: tuple[int, int], trt_state,
                  device: torch.device) -> np.ndarray:

    # Prepare input data
    org_h, org_w = img.shape[:2]
    img_input = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    img_input = img_input.astype(np.float32).transpose(2, 0, 1)[None, ...] / 255

    # Copy the data to appropriate memory
    np.copyto(trt_state['inputs'][0].host, img_input.ravel())

    # Feed to model
    heatmaps = trt_state['inf_helper']()[0]
    heatmaps = heatmaps.reshape(1, 25, img_input.shape[2] // 4, img_input.shape[3] // 4)
    points, prob = keypoints_from_heatmaps(heatmaps=heatmaps,
                                           center=np.array([[org_w // 2,
                                                             org_h // 2]]),
                                           scale=np.array([[org_w, org_h]]),
                                           unbiased=True, use_udp=True)

    points = np.concatenate([points[:, :, ::-1], prob], axis=2)
    return points


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='examples/sample.jpg',
                        help='image or video path')
    parser.add_argument('--output-path', type=str, default='', help='output path')
    parser.add_argument('--model', type=str, required=True, help='ckpt path')
    parser.add_argument('--model-name', type=str, required=False,
                        help='[s: ViT-S, b: ViT-B, l: ViT-L, h: ViT-H]')
    parser.add_argument('--yolo-size', type=int, required=False, default=320, help='YOLOv5 image size during inference')
    parser.add_argument('--yolo-nano', default=False, action='store_true',
                        help='Whether to use (the very fast) yolo nano (instead of small)')
    parser.add_argument('--show', default=False, action='store_true',
                        help='preview result')
    parser.add_argument('--show-yolo', default=False, action='store_true',
                        help='preview yolo result')
    parser.add_argument('--save-img', default=False, action='store_true',
                        help='save image result')
    parser.add_argument('--save-json', default=False, action='store_true',
                        help='save json result')
    args = parser.parse_args()

    # Load Yolo
    model_name = 'yolov5n' if args.yolo_nano else 'yolov5s'
    yolo_model = model_name + ('.onnx' if has_onnx else '.pt')
    yolo = torch.hub.load("ultralytics/yolov5", "custom", yolo_model)
    yolo.classes = [0]

    # Load vitpose model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    use_onnx = args.model.endswith('.onnx')
    use_trt = args.model.endswith('.engine')

    if args.model_name is None:  # onnx / trt models do not require model_cfg specification, but we need img size
        assert use_onnx or use_trt
        args.model_name = 's'

    if args.model_name == 's':
        from configs.ViTPose_small_coco_256x192 import model as model_cfg
        from configs.ViTPose_small_coco_256x192 import data_cfg
    elif args.model_name == 'b':
        from configs.ViTPose_base_coco_256x192 import model as model_cfg
        from configs.ViTPose_base_coco_256x192 import data_cfg
    elif args.model_name == 'l':
        from configs.ViTPose_large_coco_256x192 import model as model_cfg
        from configs.ViTPose_large_coco_256x192 import data_cfg
    elif args.model_name == 'h':
        from configs.ViTPose_huge_coco_256x192 import model as model_cfg
        from configs.ViTPose_huge_coco_256x192 import data_cfg

    input_path = args.input
    ext = input_path[input_path.rfind('.'):]
    img_size = data_cfg['image_size']

    if use_onnx:
        vit_pose = onnxruntime.InferenceSession(args.model,
                                                providers=['CUDAExecutionProvider',
                                                           'CPUExecutionProvider'])
        inf_fn = inference_onnx
    elif use_trt:
        logger = trt.Logger(trt.Logger.ERROR)
        trt_runtime = trt.Runtime(logger)
        trt_engine = engine_utils.load_engine(trt_runtime, args.model)

        # This allocates memory for network inputs/outputs on both CPU and GPU
        inputs, outputs, bindings, stream = engine_utils.allocate_buffers(trt_engine)
        # Execution context is needed for inference
        context = trt_engine.create_execution_context()
        trt_inf = partial(engine_utils.do_inference,
                          context=context, bindings=bindings, inputs=inputs,
                          outputs=outputs, stream=stream)
        vit_pose = {'inputs': inputs, 'inf_helper': trt_inf}
        inf_fn = inference_trt
    else:
        vit_pose = ViTPose(model_cfg)
        vit_pose.eval()

        ckpt = torch.load(args.model, map_location='cpu')
        if 'state_dict' in ckpt:
            vit_pose.load_state_dict(ckpt['state_dict'])
        else:
            vit_pose.load_state_dict(ckpt)
        vit_pose.to(device)
        inf_fn = inference

    print(f">>> Model loaded: {args.model}")

    # Load the image / video reader
    try:  # Check if is webcam
        int(input_path)
        is_video = True
    except ValueError:
        assert os.path.isfile(input_path), 'The input file does not exist'
        is_video = input_path[input_path.rfind('.') + 1:] in ['mp4']

    wait = 0
    if is_video:
        reader = VideoReader(input_path)
        wait = 15
        if args.save_img:
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            ret, frame = cap.read()
            cap.release()
            assert ret
            assert fps > 0
            output_size = frame.shape[:2][::-1]
            save_name = os.path.basename(input_path).replace(ext, f"_result{ext}")
            out_writer = cv2.VideoWriter(os.path.join(args.output_path, save_name),
                                         cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                         fps, output_size)
    else:
        reader = [np.array(Image.open(input_path))]

    print(f'Running inference on {input_path}')
    keypoints = []
    fps_yolo = deque([], maxlen=30)
    fps_vitpose = deque([], maxlen=30)
    for ith, img in enumerate(reader):
        # First use YOLOv5 for detection
        t0 = time.time()
        results = yolo(img, size=args.yolo_size)
        res_pd = results.pandas().xyxy[0].to_numpy()
        fps_yolo.append(1 / (time.time() - t0))

        frame_keypoints = []
        for result in res_pd:
            if result[4] < 0.4:  # TODO: Confidence finetuning
                continue
            bbox = result[:4].astype(np.float64).round().astype(int)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]] + [-10, 10], 0, img.shape[1])  # slightly bigger box
            bbox[[1, 3]] = np.clip(bbox[[1, 3]] + [-10, 10], 0, img.shape[0])

            # Crop image and pad to 3/4 aspect ratio
            img_inf = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            img_inf, (left_pad, top_pad) = pad_image(img_inf, 3 / 4)

            t0 = time.time()
            k = inf_fn(img_inf, img_size, vit_pose, device)[0]
            fps_vitpose.append(1 / (time.time() - t0))

            # Transform keypoints to original image
            k[:, :2] += bbox[:2][::-1] - [top_pad, left_pad]
            frame_keypoints.append(k)
        if ith % 30 == 0:
            print(f'>>> YOLO fps: {np.mean(fps_yolo)}')
            print(f'>>> VITPOSE fps: {np.mean(fps_vitpose)}')

        keypoints.append([v.tolist() for v in frame_keypoints])  # TODO
        if args.show or args.save_img:
            if args.show_yolo:
                img = np.array(results.render())[0]

            img = np.array(img)[:, :, ::-1]  # RGB to BGR for cv2 modules
            for k in frame_keypoints:
                img = draw_points_and_skeleton(img.copy(), k,
                                               joints_dict()['coco']['skeleton'],
                                               person_index=0,
                                               points_color_palette='gist_rainbow',
                                               skeleton_color_palette='jet',
                                               points_palette_samples=10,
                                               confidence_threshold=0.4)

            if args.save_img:
                if is_video:
                    out_writer.write(img)
                else:
                    save_name = os.path.basename(input_path).replace(ext, f"_result{ext}")
                    cv2.imwrite(os.path.join(args.output_path, save_name), img)

            if args.show:
                cv2.imshow('preview', img)
                if cv2.waitKey(wait) == ord('q'):
                    break

    if args.save_json:
        print('>>> Saving output json')
        save_name = os.path.basename(input_path).replace(ext, "_result.json")
        with open(os.path.join(args.output_path, save_name), 'w') as f:
            out = {'keypoints': keypoints.tolist()}
            out['skeleton'] = joints_dict()['coco']['keypoints']
            json.dump(out, f)

    if is_video and args.save_img:
        out_writer.release()
    cv2.destroyAllWindows()