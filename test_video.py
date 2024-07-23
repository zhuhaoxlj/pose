import os
import cv2
import numpy as np
import time
from urllib.request import urlopen
from huggingface_hub import hf_hub_download
from easy_ViTPose import VitInference
from io import BytesIO
from PIL import Image

# 配置参数
MODEL_SIZE = 'b'  # 's', 'b', 'l', 'h'
YOLO_SIZE = 's'  # 's', 'n'
DATASET = 'wholebody'  # 'coco_25', 'coco', 'wholebody', 'mpii', 'aic', 'ap10k', 'apt36k'
ext = '.pth'
ext_yolo = '.pt'
MODEL_TYPE = "torch"
YOLO_TYPE = "torch"
REPO_ID = 'JunkyByte/easy_ViTPose'
FILENAME = os.path.join(MODEL_TYPE, f'{DATASET}/vitpose-' + MODEL_SIZE + f'-{DATASET}') + ext
FILENAME_YOLO = 'yolov8/yolov8' + YOLO_SIZE + ext_yolo

# 下载模型
print(f'Downloading model {REPO_ID}/{FILENAME}')
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
yolo_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME_YOLO)

model = VitInference(model_path, yolo_path, MODEL_SIZE,
                     dataset=DATASET, yolo_size=320, is_video=False)

# Load the video
video_path = './1.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Create a window
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Run inference on frame
        frame_keypoints = model.inference(frame)
        frame_with_keypoints = model.draw(show_yolo=True)

        # Display the frame
        cv2.imshow('Video', frame_with_keypoints)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()