import os
import sys
import numpy as np
import cv2
from rknn.api import RKNN

# --- 1. 配置模型和参数 ---
ONNX_MODEL = 'best.onnx'
RKNN_MODEL = 'best.rknn'
IMG_PATH = './test.jpg' # 请确保此路径有图片
DATASET = './dataset.txt' # 请确保此文件存在且至少有一行图片路径

QUANTIZE_ON = False
OBJ_THRESH = 0.25 # 置信度阈值
NMS_THRESH = 0.45 # NMS阈值 (通常建议0.45)
IMG_SIZE = 640

# --- 确保您的类别名称与训练时一致 ---
# 示例有10个类别
CLASSES = ('first', 'good', 'stop')

# --- 2. 辅助函数 ---

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def xywh2xyxy(x):
    """
    将 [center_x, center_y, width, height] 格式的边界框转换为 [x1, y1, x2, y2] 格式。
    (x1, y1) 是左上角, (x2, y2) 是右下角。
    """
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def nms_boxes(boxes, scores):
    """
    执行非极大值抑制 (NMS)。
    """
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
    keep = np.array(keep)
    return keep

def draw(image, boxes, scores, classes):
    """
    在图片上绘制边界框和标签。
    """
    for box, score, cl in zip(boxes, scores, classes):
        # 解包边界框坐标
        left, top, right, bottom = box
        print(f'class: {CLASSES[cl]}, score: {score:.2f}')
        print(f'box coordinate left,top,right,bottom: [{left:.2f}, {top:.2f}, {right:.2f}, {bottom:.2f}]')
        
        left = int(left)
        top = int(top)
        right = int(right)
        bottom = int(bottom)

        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, f'{CLASSES[cl]} {score:.2f}',
                    (left, top - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """
    将图像缩放并填充到指定尺寸，保持宽高比。
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

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


# --- 3. 核心后处理函数 (适配单输出模型) ---

def yolov5_post_process_single_output(output):
    """
    对YOLOv5的单输出张量进行后处理。
    """
    # 输出张量形状: (1, 25200, 15), 其中 15 = 4(box) + 1(conf) + 10(classes)
    
    # 去掉batch维度
    output = output[0]

    # 过滤掉置信度低于阈值的边界框
    box_conf = output[:, 4]
    mask = box_conf >= OBJ_THRESH
    
    detections = output[mask]
    if not detections.shape[0]:
        return None, None, None

    # 计算最终得分 (目标置信度 * 类别置信度)
    class_scores = detections[:, 5:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = box_conf[mask] * np.max(class_scores, axis=1)
    
    # 再次过滤，确保最终得分也高于阈值
    mask = scores >= OBJ_THRESH
    detections = detections[mask]
    class_ids = class_ids[mask]
    scores = scores[mask]

    if not detections.shape[0]:
        return None, None, None
    
    # 转换边界框格式
    boxes_xywh = detections[:, :4]
    boxes_xyxy = xywh2xyxy(boxes_xywh)

    # 对所有检测结果执行NMS
    keep = nms_boxes(boxes_xyxy, scores)
    
    if len(keep) > 0:
        return boxes_xyxy[keep], class_ids[keep], scores[keep]
    else:
        return None, None, None

# --- 4. 主程序 ---

if __name__ == '__main__':

    # 创建RKNN对象
    rknn = RKNN(verbose=True)

    # 1. 配置模型
    print('--> Config model')
    # 对于YOLOv5, 通常输入是0-255的uint8, 模型内部会做归一化
    # 如果你的ONNX模型输入是float32, 并且已经归一化到0-1, 则使用下面的配置
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3566')
    print('done')

    # 2. 加载ONNX模型
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # 3. 构建RKNN模型
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # 4. 导出RKNN模型
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # 5. 初始化运行时环境
    print('--> Init runtime environment')
    # 在PC上模拟运行时，target可以不填。在板端运行时，请指定为 'rk3566'
    ret = rknn.init_runtime() 
    # ret = rknn.init_runtime(target='rk3566')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # 6. 准备输入图像
    img_orig = cv2.imread(IMG_PATH)
    if img_orig is None:
        print(f"Error: Cannot read image from {IMG_PATH}")
        exit(-1)

    # 使用letterbox进行预处理，以保持长宽比
    img, ratio, (dw, dh) = letterbox(img_orig, new_shape=(IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 7. 模型推理
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    print('done')

    # 8. 后处理 (!!! 这是修改后的核心部分 !!!)
    output_data = outputs[0]
    boxes, classes, scores = yolov5_post_process_single_output(output_data)

    # 9. 绘制结果并保存
    img_result = img_orig.copy()
    if boxes is not None:
        # 坐标还原到原始图像尺寸
        boxes[:, 0] -= dw
        boxes[:, 2] -= dw
        boxes[:, 1] -= dh
        boxes[:, 3] -= dh
        boxes /= ratio[0]

        draw(img_result, boxes, scores, classes)
        cv2.imwrite('result.jpg', img_result)
        print("Detection result saved to result.jpg")
    else:
        print("No objects detected.")

    # 10. 释放资源
    rknn.release()