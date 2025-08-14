import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ---------------------------------------------------------------- #
# 1. 参数设置 (Parameters)
# ---------------------------------------------------------------- #
# 定义所有你想要采集的手势类别
# Define all gesture classes you want to collect
gesture_names = ['xiexie', 'wc', 'home', 'where', 'easy','welcome','eat','left','go']

# <-- 修改这里来改变要收集的手势类别
# <-- Modify here to change the gesture class to be collected
current_gesture_index = 8
gesture_type = gesture_names[current_gesture_index] 

# 数据集保存路径 (Dataset save path)
base_dir = 'dataset'
images_base_dir = os.path.join(base_dir, 'images')
labels_base_dir = os.path.join(base_dir, 'labels')

# 训练集和验证集图片数量分割点
# Split point for training and validation set
train_val_split_count = 500

# 确保主目录存在 (Ensure base directories exist)
os.makedirs(images_base_dir, exist_ok=True)
os.makedirs(labels_base_dir, exist_ok=True)

# 采集频率控制 (每秒约20次)
# Capture frequency control (approx. 20 times per second)
capture_interval = 1 / 20  # 0.05 seconds
last_capture_time = 0

# 文件命名计数器
# File naming counter
counter = 0

# ---------------------------------------------------------------- #
# 2. 初始化MediaPipe Hands (Initialize MediaPipe Hands)
# ---------------------------------------------------------------- #
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # <-- 修改为2，以检测两只手 (Changed to 2 to detect two hands)
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# ---------------------------------------------------------------- #
# 3. 初始化摄像头 (Initialize Camera)
# ---------------------------------------------------------------- #
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("错误：无法打开摄像头。")
    exit()

print(f"摄像头已启动，准备采集手势: '{gesture_type}'")
print(f"数据将保存在 '{base_dir}' 目录下。")
print(f"前 {train_val_split_count} 张图片用于训练集，之后用于验证集。")
print("按'q'键退出程序。")

# ---------------------------------------------------------------- #
# 4. 主循环 (Main Loop)
# ---------------------------------------------------------------- #
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("忽略空的摄像头帧。")
        continue

    # 原始图像，用于保存 (Keep the original frame for saving)
    original_frame = frame.copy()
    
    # 将图像从BGR转换为RGB (Convert the BGR image to RGB)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 进行手部检测 (Process the image and find hands)
    results = hands.process(image_rgb)
    
    current_time = time.time()
    
    # ---------------------------------------------------------------- #
    # 5. 数据采集逻辑 (Data Collection Logic)
    # ---------------------------------------------------------------- #
    # 如果检测到手部，并且达到了采集时间间隔
    # If hands are detected and capture interval is met
    if results.multi_hand_landmarks and (current_time - last_capture_time) >= capture_interval:
        last_capture_time = current_time
        
        all_landmarks_x = []
        all_landmarks_y = []
        
        # 遍历所有检测到的手 (Iterate over all detected hands)
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                all_landmarks_x.append(landmark.x)
                all_landmarks_y.append(landmark.y)

        if not all_landmarks_x:
             continue # 如果没有关节点，则跳过
        
        # 获取图像尺寸 (Get image dimensions)
        h, w, _ = original_frame.shape
        
        # 计算包含所有手的最大边界框
        # Calculate the max bounding box containing all hands
        xmin = max(min(all_landmarks_x) - 0.07, 0)                # 左边界：不小于0
        ymin = max(min(all_landmarks_y) - 0.07, 0)                # 上边界：不小于0
        xmax = min(max(all_landmarks_x) + 0.07, 1)            # 右边界：不大于1
        ymax = min(max(all_landmarks_y) + 0.07, 1)            # 下边界：不大于1
        
        # 转换为YOLO格式（归一化中心点和宽高）
        # Convert to YOLO format (normalized center_x, center_y, width, height)
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        box_width = xmax - xmin
        box_height = ymax - ymin
        
        # 获取手势类别的索引 (Get the class ID for the gesture)
        class_id = gesture_names.index(gesture_type)
        
        # 格式化YOLO标签字符串 (Format the YOLO label string)
        yolo_label = f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
        
        # --- 决定是保存到训练集还是验证集 ---
        # --- Decide whether to save to train or val set ---
        if counter < train_val_split_count:
            set_type = 'train'
        else:
            set_type = 'val'
            
        # 动态创建子目录 (Dynamically create subdirectories)
        current_image_dir = os.path.join(images_base_dir, set_type)
        current_label_dir = os.path.join(labels_base_dir, set_type)
        os.makedirs(current_image_dir, exist_ok=True)
        os.makedirs(current_label_dir, exist_ok=True)

        # 生成文件名 (Generate file name)
        file_name = f"{gesture_type}_{counter:05d}"
        image_path = os.path.join(current_image_dir, f"{file_name}.jpg")
        label_path = os.path.join(current_label_dir, f"{file_name}.txt")
        
        # 保存图像和标签文件 (Save image and label file)
        cv2.imwrite(image_path, original_frame)
        with open(label_path, 'w') as f:
            f.write(yolo_label)
            
        print(f"已保存到 {set_type} 集: {file_name}.jpg 和 {file_name}.txt")
        counter += 1

        # 在显示窗口中绘制边界框以提供反馈
        # Draw bounding box on the display frame for feedback
        xmin_pixel = int(xmin * w)
        ymin_pixel = int(ymin * h)
        xmax_pixel = int(xmax * w)
        ymax_pixel = int(ymax * h)
        cv2.rectangle(frame, (xmin_pixel, ymin_pixel), (xmax_pixel, ymax_pixel), (0, 255, 0), 2)
        cv2.putText(frame, f"Capturing for '{set_type}': {gesture_type}", (xmin_pixel, ymin_pixel - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 翻转图像以获得镜像效果 (Flip the frame horizontally for a selfie-view)
    flipped_frame = cv2.flip(frame, 1)
    
    # 显示实时预览 (Display the live preview)
    cv2.imshow(f'Capturing "{gesture_type}"', flipped_frame)

    # 按'q'退出 (Press 'q' to quit)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# ---------------------------------------------------------------- #
# 6. 释放资源 (Release Resources)
# ---------------------------------------------------------------- #
hands.close()
cap.release()
cv2.destroyAllWindows()
print("程序结束，资源已释放。")