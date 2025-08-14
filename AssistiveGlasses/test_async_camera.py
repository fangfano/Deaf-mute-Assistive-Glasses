import numpy as np
import cv2
from rknnlite.api import RKNNLite
import time
import threading

# --- 1. 参数定义 ---
RKNN_MODEL = 'best.rknn'
IMG_SIZE = 640  # 必须为640，以匹配模型输入
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
CLASSES = ('first', 'good', 'stop') # 请根据您的模型修改类别

# --- 2. 异步摄像头读取类 ---
class Camera:
    """
    一个使用独立线程进行摄像头画面异步读取的类，
    以避免I/O操作阻塞主程序的推理流程。
    """
    def __init__(self, device_id=0):
        print("正在初始化摄像头...")
        self.cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        
        # 严格按照要求设置摄像头分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            raise RuntimeError("错误：无法打开摄像头。请检查设备是否连接或设备ID是否正确。")
        
        # 获取并打印实际生效的分辨率
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"摄像头已启动，请求分辨率: 640x480, 实际生效分辨率: {int(actual_width)}x{int(actual_height)}")


        # 使用线程锁来确保在读写帧时的线程安全
        self.lock = threading.Lock()
        self.frame = None
        
        # 启动后台更新线程
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True # 设置为守护线程，主程序退出时线程也退出
        self.thread.start()
        print("摄像头异步读取线程已启动。")

    def _update(self):
        """线程的目标函数，循环从摄像头硬件读取最新帧。"""
        while True:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                # 如果读取失败，可以等待一小会
                time.sleep(0.01)

    def read(self):
        """
        从主线程调用此方法来获取最新帧。
        返回帧的副本以避免多线程数据冲突。
        """
        frame_copy = None
        with self.lock:
            if self.frame is not None:
                frame_copy = self.frame.copy()
        return frame_copy

    def release(self):
        """释放摄像头资源。"""
        self.cap.release()
        print("摄像头资源已释放。")

# --- 3. 辅助函数 (YOLO后处理) ---
def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2; y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2; y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def nms_boxes(boxes, scores):
    x = boxes[:, 0]; y = boxes[:, 1]; w = boxes[:, 2] - x; h = boxes[:, 3] - y
    areas = w * h; order = scores.argsort()[::-1]; keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]]); yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]]); yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001); h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1; ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]; order = order[inds + 1]
    return np.array(keep)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

def post_process_single_output(outputs):
    if outputs is None: return None, None, None
    output = np.squeeze(outputs[0])
    if output.ndim == 1: output = np.expand_dims(output, axis=0)
    box_conf = output[:, 4]
    mask = box_conf >= OBJ_THRESH
    detections = output[mask]
    if not detections.shape[0]: return None, None, None
    class_scores = detections[:, 5:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = detections[:, 4] * np.max(class_scores, axis=1)
    mask = scores >= OBJ_THRESH
    detections = detections[mask]; class_ids = class_ids[mask]; scores = scores[mask]
    if not detections.shape[0]: return None, None, None
    boxes = xywh2xyxy(detections[:, :4])
    keep_indices = nms_boxes(boxes, scores)
    return boxes[keep_indices], class_ids[keep_indices], scores[keep_indices]

# --- 5. 主程序 ---
if __name__ == '__main__':
    rknn_lite = RKNNLite()
    camera = None  # 初始化camera变量

    try:
        print('--> 加载 RKNN 模型')
        ret = rknn_lite.load_rknn(RKNN_MODEL)
        if ret != 0: raise RuntimeError(f'加载 RKNN 模型失败, ret={ret}')
        print('完成')

        print('--> 初始化运行时环境')
        # 严格按照要求使用 init_runtime()
        ret = rknn_lite.init_runtime()
        if ret != 0: raise RuntimeError(f'初始化运行时环境失败, ret={ret}')
        print('完成')

        # 实例化并启动异步摄像头
        camera = Camera(0)
        # 等待1秒，确保摄像头线程已经成功捕获到第一帧
        time.sleep(1.0)

        print("\n程序已启动，正在进行实时检测...")
        print("在终端按 Ctrl+C 即可退出程序。")
        
        frame_count = 0
        start_time = time.time()

        while True:
            # 从异步读取器获取最新帧，此操作几乎不耗时
            frame_orig = camera.read()
            if frame_orig is None:
                # 如果初始几帧没读到，短暂等待
                time.sleep(0.01)
                continue

            # 图像预处理
            frame, _, _ = letterbox(frame_orig, new_shape=(IMG_SIZE, IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # NPU推理
            outputs = rknn_lite.inference(inputs=[frame])

            # 结果后处理
            boxes, classes, scores = post_process_single_output(outputs)

            # 处理检测结果
            if boxes is not None:
                # 为了测量纯粹的推理速度，这里可以只做简单打印
                # print(f"检测到 {len(boxes)} 个目标。")
                pass # 保持为空以进行最快的性能测试
                        
            frame_count += 1
            
            # 每100帧计算并打印一次平均FPS，避免刷屏
            if frame_count > 0 and frame_count % 100 == 0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                if elapsed_time > 0:
                    fps = frame_count / elapsed_time
                    print(f"--- 100帧内平均处理速度: {fps:.2f} FPS ---")
                    # 重置计数器以计算下一个100帧的FPS
                    frame_count = 0
                    start_time = time.time()

    except Exception as e:
        print(f"\n程序发生错误: {e}")
    except KeyboardInterrupt:
        print("\n程序被用户中断。")
    finally:
        print('--> 正在释放资源...')
        if camera:
            camera.release()
        rknn_lite.release()
        print('完成')