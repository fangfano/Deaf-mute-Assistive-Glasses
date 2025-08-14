import torch
import cv2
import numpy as np
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# ---------------------------------------------------------------- #
# 1. 加载模型
# ---------------------------------------------------------------- #
# 模型路径 (替换为你自己的 best.pt 路径)
import os
model_path = 'best.pt' # 假设在 yolov5 目录下运行
#model_path = 'yolov5s.pt' # 假设在 yolov5 目录下运行

# 从 PyTorch Hub 加载自定义模型
# 'custom' 表示加载本地的自定义训练模型
# path_or_model 指定权重的路径
# source='local' 表示使用本地的 YOLOv5 仓库代码
try:
    model = torch.hub.load('.', 'custom', path=model_path, source='local', force_reload=True)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure you are in the 'yolov5' directory and the model path is correct.")
    exit()

# 设置置信度阈值
model.conf = 0.25  # NMS confidence threshold
# 设置IoU阈值
model.iou = 0.25  # NMS IoU threshold

# ---------------------------------------------------------------- #
# 2. 初始化摄像头
# ---------------------------------------------------------------- #
# '0' 代表默认摄像头。如果有多个摄像头，可以尝试 '1', '2' 等
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

print("Camera started. Press 'q' to quit.")


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 使用常见的480高度

# 读取并打印实际的分辨率，验证设置是否生效
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Camera started with actual resolution: {int(width)}x{int(height)}")



# ---------------------------------------------------------------- #
# 3. 实时处理视频流
# ---------------------------------------------------------------- #
while True:
    # 读取一帧
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # ---------------------------------------------------------------- #
    # 4. 进行推理
    # ---------------------------------------------------------------- #
    # 将图像帧传入模型
    results = model(frame)

    # ---------------------------------------------------------------- #
    # 5. 解析结果并可视化
    # ---------------------------------------------------------------- #
    # results.pandas().xyxy[0] 会返回一个包含检测结果的 DataFrame
    # 格式: xmin, ymin, xmax, ymax, confidence, class, name
    detections = results.pandas().xyxy[0]

    # 遍历所有检测到的物体
    for index, row in detections.iterrows():
        # 获取坐标、置信度和类别名称
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row['confidence']
        name = row['name']
        
        # 在图像上绘制边界框
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # 准备要显示的标签文本
        label = f"{name}: {confidence:.2f}"
        
        # 在边界框上方绘制标签背景
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (xmin, ymin - 20), (xmin + w, ymin), (0, 255, 0), -1)
        
        # 在图像上绘制标签文本
        cv2.putText(frame, label, (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


    # 显示结果帧
    cv2.imshow('YOLOv5 Hand Gesture Recognition', frame)

    # 检测按键，如果按下 'q'，则退出循环
    if cv2.waitKey(1) == ord('q'):
        break

# ---------------------------------------------------------------- #
# 6. 释放资源
# ---------------------------------------------------------------- #
cap.release()
cv2.destroyAllWindows()
print("Resources released.")