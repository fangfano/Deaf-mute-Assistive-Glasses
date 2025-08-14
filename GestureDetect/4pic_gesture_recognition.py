import torch
import cv2
import numpy as np
import os
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
# ---------------------------------------------------------------- #
# 1. 加载模型
# ---------------------------------------------------------------- #
# 模型路径 (请确保 'best.pt' 文件在此脚本所在的目录中)
model_path = 'best.pt'

# 检查模型文件是否存在
if not os.path.exists(model_path):
    print(f"错误: 模型文件未找到，路径为: '{model_path}'")
    print("请将 'best.pt' 文件放置在与此脚本相同的目录中。")
    exit()

# 从 PyTorch Hub 加载自定义模型
# 'custom' 表示加载本地的自定义训练模型
# path_or_model 指定权重的路径
# source='local' 表示使用本地的 YOLOv5 仓库代码
try:
    # 假设yolov5的源码在当前目录下或者Python路径可寻找到
    # 如果yolov5文件夹不在当前目录，需要提供其路径，例如：torch.hub.load('path/to/yolov5', 'custom', ...)
    model = torch.hub.load('.', 'custom', path=model_path, source='local')
except Exception as e:
    print(f"加载模型时出错: {e}")
    print("请确保您在 'yolov5' 目录下，并且模型路径是正确的。")
    exit()

# 设置置信度阈值
model.conf = 0.25  # NMS 置信度阈值
# 设置IoU阈值
model.iou = 0.45  # NMS IoU 阈值

# ---------------------------------------------------------------- #
# 2. 加载图片
# ---------------------------------------------------------------- #
image_path = 'test.jpg'

# 检查图片文件是否存在
if not os.path.exists(image_path):
    print(f"错误: 图片文件未找到，路径为: '{image_path}'")
    print("请将 'test.jpg' 文件放置在与此脚本相同的目录中。")
    exit()

# 使用OpenCV读取图片
frame = cv2.imread(image_path)

if frame is None:
    print(f"错误: 无法读取图片: '{image_path}'")
    exit()

print(f"图片 '{image_path}' 加载成功。")

# ---------------------------------------------------------------- #
# 3. 进行推理
# ---------------------------------------------------------------- #
# 将图像帧传入模型
results = model(frame)

# ---------------------------------------------------------------- #
# 4. 解析结果并可视化
# ---------------------------------------------------------------- #
# results.pandas().xyxy[0] 会返回一个包含检测结果的 DataFrame
# 格式: xmin, ymin, xmax, ymax, confidence, class, name
detections = results.pandas().xyxy[0]

print(f"在图片中检测到 {len(detections)} 个目标。")

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

# ---------------------------------------------------------------- #
# 5. 保存结果
# ---------------------------------------------------------------- #
output_path = 'test_result.jpg'
cv2.imwrite(output_path, frame)

print(f"处理完成，结果已保存为 '{output_path}'。")

# 可选：显示结果图片
# cv2.imshow('YOLOv5 Detection Result', frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()