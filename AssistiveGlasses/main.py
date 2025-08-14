# coding: utf-8
import threading
import multiprocessing
import queue
import time
import cv2
import numpy as np
from rknnlite.api import RKNNLite
import tkinter as tk
import pygame
import os
import json
import sys
import subprocess
import uuid # (MODIFIED) 引入uuid库来生成唯一ID

# --- REVISED: Baidu ASR imports ---
import websocket
import requests
import base64
import hashlib
import hmac
import ssl

# --- 1. Configuration ---
# -- Gesture Recognition Config --
RKNN_MODEL = 'best.rknn'
IMG_SIZE = 640
OBJ_THRESH = 0.3
NMS_THRESH = 0.45
CLASSES = ('xiexie', 'wc', 'home', 'where', 'easy','welcome','eat','left','go')
CAM_DEVICE_ID = 0
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_BUFFER_SIZE = 1

# -- Audio & UI Config --
GESTURE_ACTIONS = {
    'xiexie': {'text': '谢谢你', 'audio': 'xiexie.wav'},
    'wc':     {'text': '去一下厕所', 'audio': 'wc.wav'},
    'home':   {'text': '回家', 'audio': 'home.wav'},
    'where':  {'text': '在哪儿', 'audio': 'where.wav'},
    'easy':   {'text': '没关系', 'audio': 'easy.wav'},
    'welcome':{'text': '不客气', 'audio': 'welcome.wav'},
    'eat':    {'text': '吃饭', 'audio': 'eat.wav'},
    'left':   {'text': '请左拐', 'audio': 'left.wav'},
    'go':     {'text': '请直行', 'audio': 'go.wav'}
}
ACTION_COOLDOWN = 3
SHOW_PARTIAL_RESULTS = True
AUDIO_CAPTURE_DEVICE_ID = 2
AUDIO_SAMPLE_RATE = 16000 # 百度要求 16000 或 8000
AUDIO_BLOCK_SIZE = 1280   # 百度建议每40ms发送一次, 16000 * 0.040 * 2 (bytes/sample) = 1280

# --- NEW: Baidu ASR Configuration (必须填写!) ---
BAIDU_APP_ID = "11965976"           # 你的 APP ID
BAIDU_API_KEY = "isIfSiKxDRj5aKGtfyxBl75Y"       # 你的 API Key
BAIDU_SECRET_KEY = "zCTSrmnrScT4UztNL1pCCd3Zz7TcDxjL"    # 你的 Secret Key
BAIDU_DEV_PID = 1537              # 1537: 普通话(纯中文识别)

# --- GPIO Configuration (NEW) ---
WPI_PIN_NUMBER = 5          # 您要监测的 wPi 引脚编号 (来自`gpio readall`的 wPi 列)
GPIO_CHECK_INTERVAL = 0.5   # 检查 GPIO 状态的间隔时间（秒）


# --- 2. Process-Safe Queues and Events ---
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)
gesture_audio_queue = queue.Queue(maxsize=1)
gesture_text_queue = queue.Queue(maxsize=5)
audio_status_queue = multiprocessing.Queue() # (MODIFIED) 新增：用于音频状态通信的队列
stop_event = multiprocessing.Event()
audio_playing_event = multiprocessing.Event()
audio_data_queue = multiprocessing.Queue()
speech_result_queue = multiprocessing.Queue()
speech_recognition_enabled_event = multiprocessing.Event()


# --- 3. Helper Functions (No changes) ---
def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def nms_boxes(boxes, scores):
    x, y = boxes[:, 0], boxes[:, 1]
    w, h = boxes[:, 2] - x, boxes[:, 3] - y
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
        w1 = np.maximum(0.0, xx2 - xx1 + 1e-5)
        h1 = np.maximum(0.0, yy2 - yy1 + 1e-5)
        inter = w1 * h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    return np.array(keep)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
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

def post_process_single_output(outputs):
    output = np.squeeze(outputs[0])
    if len(output.shape) == 1:
        return None, None, None
    box_conf = output[:, 4]
    mask = box_conf >= OBJ_THRESH
    detections = output[mask]
    if not detections.shape[0]:
        return None, None, None
    class_scores = detections[:, 5:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = detections[:, 4] * np.max(class_scores, axis=1)
    mask = scores >= OBJ_THRESH
    detections, class_ids, scores = detections[mask], class_ids[mask], scores[mask]
    if not detections.shape[0]:
        return None, None, None
    boxes = xywh2xyxy(detections[:, :4])
    keep_indices = nms_boxes(boxes, scores)
    return boxes[keep_indices], class_ids[keep_indices], scores[keep_indices]

def check_gpio_status():
    command = ["gpio", "read", str(WPI_PIN_NUMBER)]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=2
        )
        return result.stdout.strip()
    except FileNotFoundError:
        print("\n[GPIO 错误] 'gpio' 命令未找到。请确认 WiringOP 已安装且在系统 PATH 中。", file=sys.stderr)
        return None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"\n[GPIO 错误] 命令 '{' '.join(command)}' 执行失败: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"\n[GPIO 未知错误] 发生未知错误: {e}", file=sys.stderr)
        return None

def gpio_control_thread(enabled_ev, stop_ev):
    print("[Thread] GPIO control thread started.")
    print(f">>> 正在监控 wPi 引脚 {WPI_PIN_NUMBER} 以控制语音识别。<<<")
    print(">>> 高电平 -> 开启转录 | 低电平 -> 关闭转录 <<<")
    last_known_state = -1
    while not stop_ev.is_set():
        pin_value = check_gpio_status()
        if pin_value == "1":
            if last_known_state != 1:
                print("[GPIO 控制] 状态变化: 高电平 -> 开启语音识别")
                enabled_ev.set()
                last_known_state = 1
        elif pin_value == "0":
            if last_known_state != 0:
                print("[GPIO 控制] 状态变化: 低电平 -> 关闭语音识别")
                enabled_ev.clear()
                last_known_state = 0
        else:
            if last_known_state != -1:
                 print("[GPIO 控制] GPIO读取失败，维持当前状态。")
            last_known_state = -1
        time.sleep(GPIO_CHECK_INTERVAL)
    print("[Thread] GPIO control thread finished.")

def camera_thread_func():
    print("[Thread] Camera thread started.")
    cap = cv2.VideoCapture(CAM_DEVICE_ID)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        stop_event.set()
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, CAM_BUFFER_SIZE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    print(f"Camera started with resolution: {int(cap.get(3))}x{int(cap.get(4))}")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass
    cap.release()
    print("[Thread] Camera thread finished.")

def inference_thread_func():
    print("[Thread] Inference thread started.")
    rknn_lite = RKNNLite()
    if rknn_lite.load_rknn(RKNN_MODEL) != 0:
        print(f"Error: Failed to load RKNN model: {RKNN_MODEL}")
        stop_event.set()
        return
    if rknn_lite.init_runtime() != 0:
        print("Error: Failed to initialize runtime.")
        stop_event.set()
        return
    print("Inference engine ready.")
    while not stop_event.is_set():
        try:
            frame_orig = frame_queue.get(timeout=1)
            frame, ratio, (dw, dh) = letterbox(frame_orig, new_shape=(IMG_SIZE, IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            outputs = rknn_lite.inference(inputs=[frame])
            try:
                result_queue.put_nowait({
                    "original_frame": frame_orig,
                    "outputs": outputs,
                    "ratio_pad": (ratio, (dw, dh))
                })
            except queue.Full:
                pass
        except queue.Empty:
            continue
    rknn_lite.release()
    print("[Thread] Inference thread finished.")

def post_processing_thread_func():
    print("[Thread] Post-processing thread started.")
    prev_time = time.time()
    last_action_time = {}
    while not stop_event.is_set():
        try:
            data = result_queue.get(timeout=1)
            frame_orig = data["original_frame"]
            outputs = data["outputs"]
            ratio, (dw, dh) = data["ratio_pad"]
            boxes, classes, scores = post_process_single_output(outputs)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            if boxes is not None:
                boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio[0]
                boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio[1]
                for box, score, cl_id in zip(boxes, scores, classes):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = CLASSES[cl_id]
                    cv2.rectangle(frame_orig, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame_orig, f'{class_name} {score:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    if class_name in GESTURE_ACTIONS:
                        now = time.time()
                        if now - last_action_time.get(class_name, 0) > ACTION_COOLDOWN:
                            last_action_time[class_name] = now
                            action = GESTURE_ACTIONS[class_name]
                            try:
                                # (MODIFIED) 增加唯一ID，用于追踪UI和音频
                                event_id = str(uuid.uuid4())
                                
                                while not gesture_audio_queue.empty():
                                    try:
                                        gesture_audio_queue.get_nowait()
                                    except queue.Empty:
                                        continue
                                
                                # (MODIFIED) 将包含ID的字典放入队列
                                gesture_audio_queue.put_nowait({
                                    'id': event_id,
                                    'audio': action['audio']
                                })
                                gesture_text_queue.put_nowait({
                                    'type': 'gesture',
                                    'id': event_id,
                                    'text': action['text']
                                })
                            except queue.Full:
                                pass
            cv2.putText(frame_orig, f"FPS: {fps:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
        except queue.Empty:
            continue
    cv2.destroyAllWindows()
    print("[Thread] Post-processing thread finished.")

# (MODIFIED) 改造音频播放线程以发送状态更新
def audio_playback_thread_func():
    print("[Thread] Audio playback thread started.")
    pygame.mixer.init()
    current_audio_info = None

    while not stop_event.is_set():
        try:
            # 检查是否有新的抢占任务
            audio_info = gesture_audio_queue.get_nowait()
            audio_file = audio_info['audio']
            event_id = audio_info['id']

            if os.path.exists(audio_file):
                # 如果有旧的音频在播放，通知UI它的播放已“结束”（被中断）
                if current_audio_info and current_audio_info.get('id'):
                    audio_status_queue.put({'id': current_audio_info['id'], 'status': 'finished'})
                
                print(f"[Audio] 收到新任务，中断并播放: {audio_file}")
                current_audio_info = audio_info
                audio_playing_event.set()
                pygame.mixer.music.stop()
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                # 通知UI，新的音频已开始播放
                audio_status_queue.put({'id': event_id, 'status': 'playing'})
            else:
                print(f"[Audio] 文件不存在: {audio_file}")

        except queue.Empty:
            # 队列为空，检查当前音频是否播放完毕
            if current_audio_info and not pygame.mixer.music.get_busy():
                if audio_playing_event.is_set():
                    # 通知UI，当前音频已正常播放结束
                    audio_status_queue.put({'id': current_audio_info['id'], 'status': 'finished'})
                    print(f"[Audio] 文件 {current_audio_info['audio']} 播放结束.")
                    current_audio_info = None
                    audio_playing_event.clear()
            
            time.sleep(0.05)
            
        except Exception as e:
            print(f"Audio Playback Error: {e}")
            if current_audio_info:
                audio_status_queue.put({'id': current_audio_info['id'], 'status': 'finished'})
            current_audio_info = None
            audio_playing_event.clear()

    pygame.mixer.quit()
    print("[Thread] Audio playback thread finished.")


def speech_capture_process_func(audio_q, stop_ev, audio_playing_ev):
    import sounddevice as sd
    import psutil
    try:
        p = psutil.Process()
        p.cpu_affinity([2])
        print(f"[Process] Speech capture process 成功绑定到CPU核心: {p.cpu_affinity()}")
    except Exception as e:
        print(f"[Process] 警告: 语音采集进程CPU亲和性设置失败: {e}")
    def audio_callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        if not audio_playing_ev.is_set():
            try:
                audio_q.put_nowait(bytes(indata))
            except queue.Full:
                pass
    try:
        stream = sd.InputStream(
            device=AUDIO_CAPTURE_DEVICE_ID,
            samplerate=AUDIO_SAMPLE_RATE,
            channels=1,
            dtype='int16',
            blocksize=AUDIO_BLOCK_SIZE,
            callback=audio_callback
        )
        with stream:
            print(f"Microphone (ID: {AUDIO_CAPTURE_DEVICE_ID}) stream started.")
            stop_ev.wait()
    except Exception as e:
        print(f"Error starting sound stream: {e}")
    print("[Process] Speech capture process finished.")

class BaiduASRClient:
    def __init__(self, app_id, api_key, dev_pid, result_q):
        self.app_id = app_id
        self.api_key = api_key
        self.dev_pid = dev_pid
        self.result_q = result_q
        self.ws = None
        self.ws_thread = None
        self.connection_opened = threading.Event()

    def _get_ws_url(self):
        sn = str(uuid.uuid4())
        base_url = "wss://vop.baidu.com/realtime_asr"
        print(f"[Baidu ASR] 准备连接到新地址: {base_url}, sn={sn}")
        return f"{base_url}?sn={sn}"

    def _on_message(self, ws, message):
        try:
            msg_obj = json.loads(message)
            err_no = msg_obj.get("err_no", -1)
            if err_no != 0:
                print(f"[Baidu ASR] 收到错误: code={err_no}, message={msg_obj.get('err_msg')}")
                return
            if msg_obj.get('type') == 'FIN_TEXT':
                self.result_q.put({'type': 'final', 'text': msg_obj.get('result', '')})
            elif msg_obj.get('type') == 'MID_TEXT' and SHOW_PARTIAL_RESULTS:
                self.result_q.put({'type': 'partial', 'text': msg_obj.get('result', '')})
        except Exception as e:
            print(f"[Baidu ASR] 消息处理错误: {e}, 收到消息: {message}")

    def _on_error(self, ws, error):
        print(f"[Baidu ASR] WebSocket 错误: {error}")
        self.connection_opened.set()

    def _on_close(self, ws, close_status_code, close_msg):
        print("[Baidu ASR] WebSocket 连接已关闭")
        self.connection_opened.set()

    def _on_open(self, ws):
        print("[Baidu ASR] WebSocket 连接成功")
        self.send_start_params()
        self.connection_opened.set()

    def connect(self):
        ws_url = self._get_ws_url()
        if not ws_url:
            return False
        self.connection_opened.clear()
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        finished_in_time = self.connection_opened.wait(timeout=10)
        if not finished_in_time:
            print("[Baidu ASR] WebSocket 连接超时")
            self.close()
            return False
        return self.ws and self.ws.sock and self.ws.sock.connected

    def send_start_params(self):
        start_params = {
            "type": "START",
            "data": {
                "appid": int(self.app_id),
                "appkey": self.api_key,
                "dev_pid": self.dev_pid,
                "cuid": "your_custom_user_id_from_python",
                "format": "pcm",
                "sample": AUDIO_SAMPLE_RATE
            }
        }
        try:
            print(f"[Baidu ASR] 发送启动参数: {json.dumps(start_params)}")
            self.ws.send(json.dumps(start_params))
        except Exception as e:
            print(f"[Baidu ASR] 发送开始参数失败: {e}")

    def send_audio(self, data):
        if self.ws and self.ws.sock and self.ws.sock.connected:
            try:
                self.ws.send(data, websocket.ABNF.OPCODE_BINARY)
            except Exception as e:
                print(f"[Baidu ASR] 发送音频数据失败: {e}")

    def send_finish(self):
        finish_params = {"type": "FINISH"}
        if self.ws and self.ws.sock and self.ws.sock.connected:
            try:
                self.ws.send(json.dumps(finish_params))
                print("[Baidu ASR] 发送结束指令")
            except Exception as e:
                print(f"[Baidu ASR] 发送结束参数失败: {e}")

    def close(self):
        if self.ws:
            self.ws.close()
        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=1)

def speech_recognition_process_func(audio_q, result_q, enabled_ev, stop_ev):
    import psutil
    try:
        p = psutil.Process()
        p.cpu_affinity([3])
        print(f"[Process] Speech recognition process 成功绑定到CPU核心: {p.cpu_affinity()}")
    except Exception as e:
        print(f"[Process] 警告: 语音识别进程CPU亲和性设置失败: {e}")
    print("[Process] Speech recognition process started.")
    if BAIDU_API_KEY == "isIfSiKxDRj5aKGtfyxBl75Y" or BAIDU_SECRET_KEY == "zCTSrmnrScT4UztNL1pCCd3Zz7TcDxjL":
        print("\n" + "="*50)
        print("警告: 您正在使用示例的百度云 Key, 可能随时失效。")
        print("建议替换为您自己的 APP_ID, API_KEY 和 SECRET_KEY!")
        print("="*50 + "\n")
    while not stop_ev.is_set():
        print("[Baidu ASR] 等待 GPIO 信号开启...")
        enabled_ev.wait()
        if stop_ev.is_set(): break
        print("[Baidu ASR] GPIO 信号已收到，正在启动识别会话...")
        asr_client = BaiduASRClient(BAIDU_APP_ID, BAIDU_API_KEY, BAIDU_DEV_PID, result_q)
        if not asr_client.connect():
            print("[Baidu ASR] 连接失败，将在5秒后重试")
            time.sleep(5)
            continue
        while enabled_ev.is_set() and not stop_ev.is_set():
            try:
                audio_data = audio_q.get(timeout=0.1)
                asr_client.send_audio(audio_data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Baidu ASR] 识别循环中发生错误: {e}")
                break
        print("[Baidu ASR] GPIO 信号已关闭或程序正在退出，正在结束识别会话...")
        asr_client.send_finish()
        time.sleep(1)
        asr_client.close()
    print("[Process] Speech recognition process finished.")

# (MODIFIED) UI线程函数签名增加 audio_status_q 参数
def ui_thread_func(speech_q, gesture_q, audio_status_q, stop_ev):
    print("[Thread] UI thread started.")
    root = tk.Tk()
    root.title("System Log")
    root.geometry("700x600+150+100")
    root.overrideredirect(True)
    root.wm_attributes("-topmost", 1)
    root.configure(bg='black')

    large_font = ("Helvetica", 28)
    text_widget = tk.Text(root, font=large_font, fg="white", bg="black",
                          borderwidth=0, highlightthickness=0, wrap=tk.WORD)
    text_widget.pack(expand=True, fill='both', padx=15, pady=15)
    text_widget.tag_configure("final", foreground="lime green")
    text_widget.tag_configure("partial", foreground="red")
    text_widget.tag_configure("gesture", foreground="cyan", font=("Helvetica", 28, "bold"))
    text_widget.config(state=tk.DISABLED)

    def check_queues():
        text_widget.config(state=tk.NORMAL)
        
        # (MODIFIED) 增加处理手势文本和ID的逻辑
        try:
            while not gesture_q.empty():
                data = gesture_q.get_nowait()
                if data.get('type') == 'gesture':
                    text = data.get('text', '').strip()
                    event_id = data.get('id')
                    if text and event_id:
                        # 使用唯一ID创建tag
                        tag_name = f"gesture_{event_id}"
                        text_widget.insert(tk.END, text + '\n', ("gesture", tag_name))
        except queue.Empty:
            pass

        # (MODIFIED) 增加处理音频状态的逻辑
        try:
            while not audio_status_q.empty():
                data = audio_status_q.get_nowait()
                event_id = data.get('id')
                status = data.get('status')
                tag_name = f"gesture_{event_id}"
                
                # 检查tag是否存在
                if text_widget.tag_ranges(tag_name):
                    start, end = text_widget.tag_ranges(tag_name)
                    line_text = text_widget.get(start, end).strip()

                    if status == 'playing':
                        new_text = f"{line_text} 【播放中】\n"
                        text_widget.delete(start, end)
                        text_widget.insert(start, new_text, ("gesture", tag_name))
                    elif status == 'finished':
                        # 从原始文本中移除【播放中】（如果存在）
                        line_text = line_text.replace("【播放中】", "").strip()
                        new_text = f"{line_text} 【播放结束】\n"
                        text_widget.delete(start, end)
                        text_widget.insert(start, new_text, ("gesture", tag_name))
        except queue.Empty:
            pass

        try:
            while not speech_q.empty():
                data = speech_q.get_nowait()
                result_type = data.get('type')
                text = data.get('text', '').strip()
                if not text: continue
                partial_ranges = text_widget.tag_ranges("partial")
                if partial_ranges:
                    text_widget.delete(partial_ranges[-2], partial_ranges[-1])
                if result_type == 'final':
                    text_widget.insert(tk.END, text + '\n', "final")
                elif result_type == 'partial':
                    text_widget.insert(tk.END, text, "partial")
        except queue.Empty:
            pass

        text_widget.see(tk.END)
        text_widget.config(state=tk.DISABLED)

        if not stop_ev.is_set():
            root.after(100, check_queues)
        else:
            root.destroy()

    root.after(100, check_queues)
    root.mainloop()
    stop_event.set()
    print("[Thread] UI thread finished.")


# --- 5. Main Execution ---
if __name__ == '__main__':
    multiprocessing.freeze_support()
    import psutil

    try:
        p = psutil.Process()
        p.cpu_affinity([0, 1])
        print(f"[Main] 主进程成功绑定到CPU核心: {p.cpu_affinity()}")
    except Exception as e:
        print(f"[Main] 警告: 设置主进程CPU亲和性失败: {e}")

    threads = [
        threading.Thread(target=camera_thread_func, daemon=True),
        threading.Thread(target=inference_thread_func, daemon=True),
        threading.Thread(target=post_processing_thread_func, daemon=True),
        threading.Thread(target=audio_playback_thread_func, daemon=True),
        threading.Thread(
            target=gpio_control_thread,
            args=(speech_recognition_enabled_event, stop_event),
            daemon=True
        ),
    ]

    # (MODIFIED) UI线程的参数增加了 audio_status_queue
    ui_thread = threading.Thread(
        target=ui_thread_func,
        args=(speech_result_queue, gesture_text_queue, audio_status_queue, stop_event)
    )
    
    processes = [
        multiprocessing.Process(
            target=speech_capture_process_func,
            args=(audio_data_queue, stop_event, audio_playing_event)
        ),
        multiprocessing.Process(
            target=speech_recognition_process_func,
            args=(audio_data_queue, speech_result_queue, speech_recognition_enabled_event, stop_event)
        )
    ]

    try:
        print("Starting all processes and threads...")
        for p in processes:
            p.start()
        for t in threads:
            t.start()
        
        ui_thread.start()
        ui_thread.join()
        print("\nMain: UI closed or 'q' pressed. Stop signal received.")

    except KeyboardInterrupt:
        print("\nMain: KeyboardInterrupt received. Stopping everything.")
        stop_event.set()

    finally:
        print("Main: Cleaning up resources...")
        speech_recognition_enabled_event.set()
        stop_event.set()
        
        all_threads = threads + [ui_thread]
        for t in all_threads:
            if t.is_alive():
                t.join(timeout=2)
                
        for p in processes:
            if p.is_alive():
                p.join(timeout=5)
                if p.is_alive():
                    print(f"Warning: Process {p.pid} did not exit gracefully. Terminating.")
                    p.terminate()

        print("All threads and processes have been terminated. Exiting.")