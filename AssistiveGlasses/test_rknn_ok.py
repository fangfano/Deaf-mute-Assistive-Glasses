import numpy as np
import cv2
from rknnlite.api import RKNNLite

# --- 1. 配置 ---
RKNN_MODEL = 'best.rknn'
IMG_PATH = 'test.jpg'
IMG_SIZE = 640
OBJ_THRESH = 0.25
NMS_THRESH = 0.45

# 类别名称应与训练时一致
CLASSES = ('first', 'good', 'stop')

# --- 2. 辅助函数 ---
def xywh2xyxy(x):
    """Converts nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]."""
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
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

def draw(image, boxes, scores, classes):
    """Draws the detection boxes on the image."""
    for box, score, cl in zip(boxes, scores, classes):
        # FIX: Unpack coordinates with clear x and y names
        x1, y1, x2, y2 = map(int, box)

        # Print the correct coordinates
        print(f'class: {CLASSES[cl]}, score: {score:.2f}, box: [{x1}, {y1}, {x2}, {y2}]')

        # FIX: Use the correct (x, y) order for drawing functions
        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw the label text above the box
        cv2.putText(image, f'{CLASSES[cl]} {score:.2f}',
                    (x1, y1 - 10),  # Position text at (x1, y1 - 10)
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

# --- 3. FIX: Final corrected post-processing function ---
def post_process_single_output(outputs):
    """
    Handles post-processing for a single, combined YOLOv5 output tensor.
    """
    output = outputs[0]

    # The raw output shape is (1, 25200, 15, 1).
    # np.squeeze() will remove all dimensions of size 1, resulting
    # in the correct shape: (25200, 15).
    output = np.squeeze(output)

    # Filter out boxes with low object confidence
    box_conf = output[:, 4]
    mask = box_conf >= OBJ_THRESH
    detections = output[mask]
    if not detections.shape[0]:
        return None, None, None

    # Calculate final scores (object_conf * class_conf)
    class_scores = detections[:, 5:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = detections[:, 4] * np.max(class_scores, axis=1)

    # Second filter based on the final score
    mask = scores >= OBJ_THRESH
    detections = detections[mask]
    class_ids = class_ids[mask]
    scores = scores[mask]
    if not detections.shape[0]:
        return None, None, None
    
    # Convert box format from [center_x, center_y, w, h] to [x1, y1, x2, y2]
    boxes = xywh2xyxy(detections[:, :4])

    # Perform Non-Maximum Suppression
    keep_indices = nms_boxes(boxes, scores)
    
    return boxes[keep_indices], class_ids[keep_indices], scores[keep_indices]

# --- 4. 主程序 ---
if __name__ == '__main__':
    rknn_lite = RKNNLite()

    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(RKNN_MODEL)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')

    print('--> Init runtime environment')
    ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    img_orig = cv2.imread(IMG_PATH)
    if img_orig is None:
        print(f"ERROR: Failed to read image from {IMG_PATH}")
        exit()
    
    img, ratio, (dw, dh) = letterbox(img_orig, new_shape=(IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print('--> Inference')
    outputs = rknn_lite.inference(inputs=[img])

    # --- ADD THIS DIAGNOSTIC CODE ---
    print("\n--- DIAGNOSTIC INFO ---")
    if outputs:
        output_tensor = outputs[0]
        print(f"Model output tensor shape: {output_tensor.shape}")
        if len(output_tensor.shape) == 3:
            # Shape is likely (batch, channels, proposals), e.g., (1, 85, 25200)
            num_channels = output_tensor.shape[1]
            # 5 channels are for box coordinates (4) and object confidence (1)
            num_classes = num_channels - 5
            print(f"This means your model has {num_classes} classes.")
        else:
            print("Unexpected output tensor shape.")
    print("-----------------------\n")
    # --- END DIAGNOSTIC CODE ---

    print('done')
    
    # --- 后处理 (已修改为单输出逻辑) ---
    print('--> Post process')
    boxes, classes, scores = post_process_single_output(outputs)

    img_result = img_orig.copy()
    if boxes is not None:
        # Rescale boxes from IMG_SIZE to original image size
        boxes[:, 0] -= dw
        boxes[:, 2] -= dw
        boxes[:, 1] -= dh
        boxes[:, 3] -= dh
        boxes /= ratio[0]
        
        draw(img_result, boxes, scores, classes)
    else:
        print("No objects detected.")
    
    output_path = 'result.jpg'
    print(f'--> Saving result to {output_path}')
    cv2.imwrite(output_path, img_result)
    print('done')

    rknn_lite.release()