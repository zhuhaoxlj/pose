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

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('result.mp4', fourcc, fps, (width, height))

frame_count = 0
start_time = time.time()

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame_count += 1
        
        # Run inference on frame
        frame_keypoints = model.inference(frame)
        frame_with_keypoints = model.draw(show_yolo=True)

        # Calculate and draw FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame_with_keypoints, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write the frame to the output video
        out.write(frame_with_keypoints)

        # Print frame information
        print(f"Processing frame {frame_count}, FPS: {fps:.2f}")
    else:
        break

# Release the video capture and writer objects
cap.release()
out.release()

print("Video processing completed. Result saved as result.mp4")