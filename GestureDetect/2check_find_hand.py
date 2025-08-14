# -*- coding: utf-8 -*-


import cv2
import mediapipe as mp
import numpy as np

# 初始化MediaPipe Hands解决方案
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    # 'STATIC_IMAGE_MODE'设置为False，以将输入视为视频流
    static_image_mode=False,
    # 最多检测2只手
    max_num_hands=2,
    # 模型复杂度：0或1。0速度更快，1精度更高
    model_complexity=1,
    # 最小检测置信度，低于此值则认为检测失败
    min_detection_confidence=0.5,
    # 最小跟踪置信度，低于此值则重新进行检测
    min_tracking_confidence=0.5)

# 初始化绘图工具
mp_drawing = mp.solutions.drawing_utils

# 初始化摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("错误：无法打开摄像头。")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 使用常见的480高度

# 读取并打印实际的分辨率，验证设置是否生效
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Camera started with actual resolution: {int(width)}x{int(height)}")

print("摄像头已启动，请在窗口中展示你的手。按'q'键退出。")

while cap.isOpened():
    # 读取一帧图像
    success, image = cap.read()
    if not success:
        print("忽略空的摄像头帧。")
        continue

    # 将图像从BGR转换为RGB，因为MediaPipe需要RGB格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 使用MediaPipe处理图像，进行手部检测
    results = hands.process(image_rgb)

    # 如果检测到了手部
    if results.multi_hand_landmarks:
        # 遍历检测到的每一只手
        for hand_landmarks in results.multi_hand_landmarks:
            # ------------------------------------------------------------ #
            # 1. 绘制手部关节点和连接线（可选，用于可视化）
            # ------------------------------------------------------------ #
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)

            # ------------------------------------------------------------ #
            # 2. 计算并绘制手部的边界框（定位手部位置）
            # ------------------------------------------------------------ #
            # 获取所有关节点的坐标
            all_x = [landmark.x for landmark in hand_landmarks.landmark]
            all_y = [landmark.y for landmark in hand_landmarks.landmark]
            
            # 计算边界框的左上角和右下角坐标
            # 注意：MediaPipe返回的坐标是归一化的(0到1之间)，需要乘以图像的宽高
            h, w, _ = image.shape
            xmin = int(min(all_x) * w)
            ymin = int(min(all_y) * h)
            xmax = int(max(all_x) * w)
            ymax = int(max(all_y) * h)

            # 增加一点边距，让框看起来更舒服
            padding = 20
            xmin = max(0, xmin - padding)
            ymin = max(0, ymin - padding)
            xmax = min(w, xmax + padding)
            ymax = min(h, ymax + padding)

            # 绘制边界框
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # 在边界框上方写上标签
            cv2.putText(image, 'Hand', (xmin, ymin - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


    # 翻转图像，使其看起来像镜子
    flipped_image = cv2.flip(image, 1)

    # 显示结果图像
    cv2.imshow('MediaPipe Hand Tracking', flipped_image)

    # 检测按键，如果按下 'q'，则退出循环
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# 释放资源
hands.close()
cap.release()
cv2.destroyAllWindows()
print("资源已释放。")