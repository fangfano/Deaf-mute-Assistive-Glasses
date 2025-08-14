import numpy as np
import cv2
from rknnlite.api import RKNNLite
import time # 新增：用于计算帧率

# --- 1. 参数定义 ---
RKNN_MODEL = 'yolov5s.rknn'
# IMG_PATH = 'test.jpg' # 不再需要
IMG_SIZE = 640
OBJ_THRESH = 0.25
NMS_THRESH = 0.45

# 类别名称
#CLASSES = ('first', 'good', 'stop')
CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")


# --- 2. 辅助函数 (保持不变) ---
def xywh2xyxy(x):
    """Converts nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]."""
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def nms_boxes(boxes, scores):
    """Performs Non-Maximum Suppression."""
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - x
    h = boxes[:, 3] - y
    areas = w * h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    return np.array(keep)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Resizes and pads image while meeting stride-multiple constraints."""
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

def draw(image, boxes, scores, classes, fps):
    """Draws the detection boxes and FPS on the image."""
    for box, score, cl in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # 绘制类别和置信度
        cv2.putText(image, f'{CLASSES[cl]} {score:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # 绘制 FPS
    cv2.putText(image, f"FPS: {fps:.2f}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# --- 3. 后处理函数 (保持不变) ---
def post_process_single_output(outputs):
    """Handles post-processing for a single, combined YOLOv5 output tensor."""
    output = np.squeeze(outputs[0])
    box_conf = output[:, 4]
    mask = box_conf >= OBJ_THRESH
    detections = output[mask]
    if not detections.shape[0]:
        return None, None, None
    class_scores = detections[:, 5:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = detections[:, 4] * np.max(class_scores, axis=1)
    mask = scores >= OBJ_THRESH
    detections = detections[mask]
    class_ids = class_ids[mask]
    scores = scores[mask]
    if not detections.shape[0]:
        return None, None, None
    boxes = xywh2xyxy(detections[:, :4])
    keep_indices = nms_boxes(boxes, scores)
    return boxes[keep_indices], class_ids[keep_indices], scores[keep_indices]

# --- 4. 主程序 ---
if __name__ == '__main__':
    rknn_lite = RKNNLite()

    print('--> 加载 RKNN 模型')
    ret = rknn_lite.load_rknn(RKNN_MODEL)
    if ret != 0:
        print('加载 RKNN 模型失败')
        exit(ret)
    print('完成')

    print('--> 初始化运行时环境')
    ret = rknn_lite.init_runtime()
    if ret != 0:
        print('初始化运行时环境失败!')
        exit(ret)
    print('完成')

    # 新增：初始化摄像头
    cap = cv2.VideoCapture(0) # 0 代表默认摄像头
    if not cap.isOpened():
        print("错误：无法打开摄像头。")
        exit()

    # 这是解决处理延迟导致画面滞后的核心所在。
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 读取并打印实际的分辨率，验证设置是否生效
    buffer_size = cap.get(cv2.CAP_PROP_BUFFERSIZE) # 确认缓冲区大小是否设置成功
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"摄像头内部缓冲区大小已设置为: {int(buffer_size)}")
    print(f"Camera started with actual resolution: {int(width)}x{int(height)}")
    
    # 新增：用于计算FPS
    prev_time = 0
    
    while True:
        # 修改：从文件读取改为循环读取摄像头帧
        ret, frame_orig = cap.read()
        if not ret:
            print("错误：无法读取视频帧。")
            break

        # 图像预处理
        frame, ratio, (dw, dh) = letterbox(frame_orig, new_shape=(IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 推理
        outputs = rknn_lite.inference(inputs=[frame])

        # 后处理
        boxes, classes, scores = post_process_single_output(outputs)
        
        # 结果可视化
        img_result = frame_orig.copy()
        
        # 计算并显示FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        if boxes is not None:
            # 将边界框坐标从 IMG_SIZE 缩放回原始图像尺寸
            boxes[:, 0] -= dw
            boxes[:, 2] -= dw
            boxes[:, 1] -= dh
            boxes[:, 3] -= dh
            boxes /= ratio[0]
            
            draw(img_result, boxes, scores, classes, fps)
        else:
            # 即使没有检测到物体，也显示FPS
            cv2.putText(img_result, f"FPS: {fps:.2f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 修改：将结果保存为图片改为实时显示
        cv2.imshow('RKNN Real-time Detection', img_result)

        # 新增：按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 新增：释放资源
    print('--> 释放资源')
    cap.release()
    cv2.destroyAllWindows()
    rknn_lite.release()
    print('完成')