# -*- coding: utf-8 -*-
"""
Vosk 多进程优化版 - 独立语音识别性能测试程序

架构:
- 主进程: 负责音频采集和结果显示，保持低CPU占用。
- 工作进程: 负责繁重的语音识别计算，会被调度到另一个CPU核心，允许其100%占用。

解决了单核CPU被打满导致的高延迟问题。
"""
import multiprocessing
import sys
import json
import os
import time

import numpy
import sounddevice as sd
import vosk

# --- 1. 配置区域 (与之前相同) ---
VOSK_MODEL_PATH = "vosk-model-small-cn-0.22" 
VOSK_DEVICE_ID = 2 
VOSK_SAMPLE_RATE = 16000
VOSK_BLOCK_SIZE = 1600 # 保持较小的 blocksize 以获得快速响应
SHOW_PARTIAL_RESULTS = True # 在此架构下，即使显示部分结果，主进程也不会卡顿

# --- 2. 工作进程函数 ---

def recognizer_process(audio_q: multiprocessing.Queue, result_q: multiprocessing.Queue):
    """
    这个函数在独立的进程中运行，专门处理语音识别。
    """
    print("[工作进程] 启动，正在加载模型...")
    try:
        model = vosk.Model(VOSK_MODEL_PATH)
        recognizer = vosk.KaldiRecognizer(model, VOSK_SAMPLE_RATE)
        recognizer.SetWords(False)
        print("[工作进程] 模型加载完毕，开始处理音频数据。")
    except Exception as e:
        print(f"[工作进程] 模型加载失败: {e}")
        return

    while True:
        # 从音频队列获取数据，如果队列中是None，则表示结束
        data = audio_q.get()
        if data is None:
            break

        if recognizer.AcceptWaveform(data):
            result_json = recognizer.Result()
            result_dict = json.loads(result_json)
            # 将最终结果放入结果队列
            result_q.put({'type': 'final', 'text': result_dict.get('text', '')})
        elif SHOW_PARTIAL_RESULTS:
            partial_json = recognizer.PartialResult()
            partial_dict = json.loads(partial_json)
            # 将部分结果放入结果队列
            result_q.put({'type': 'partial', 'text': partial_dict.get('partial', '')})
    
    print("[工作进程] 收到结束信号，正在退出。")


# --- 3. 主进程与回调 ---

def audio_callback(indata, frames, time, status):
    """音频回调函数，现在将数据放入 multiprocessing.Queue"""
    if status:
        print(status, file=sys.stderr)
    # 使用 try_put 避免在极端情况下阻塞
    try:
        audio_queue.put_nowait(bytes(indata))
    except multiprocessing.queues.Full:
        # 如果队列满了，可以忽略这个数据包，或者打印一个警告
        pass

def main():
    """主函数，负责启动和管理进程"""
    print("[主进程] 程序启动...")

    # 检查模型路径
    if not os.path.exists(VOSK_MODEL_PATH):
        print(f"[主进程] 错误: Vosk 模型路径 '{VOSK_MODEL_PATH}' 不存在。")
        sys.exit(1)

    # 创建跨进程队列
    global audio_queue
    audio_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    # 创建并启动识别工作进程
    worker = multiprocessing.Process(
        target=recognizer_process,
        args=(audio_queue, result_queue)
    )
    worker.start()

    try:
        # 创建并启动音频流
        stream = sd.InputStream(
            device=VOSK_DEVICE_ID,
            samplerate=VOSK_SAMPLE_RATE,
            channels=1,
            dtype='int16',
            blocksize=VOSK_BLOCK_SIZE,
            callback=audio_callback
        )
        stream.start()
        print(f"[主进程] 成功打开麦克风 (ID: {VOSK_DEVICE_ID})，正在采集音频...")
        print("[主进程] 按 Ctrl+C 退出。")

        # 主进程循环，只负责从结果队列中获取并显示结果
        while True:
            result = result_queue.get()
            text = result.get('text', '').strip()
            
            if not text:
                continue

            if result['type'] == 'final':
                sys.stdout.write('\r' + ' ' * 80 + '\r') 
                print(f"【最终结果】: {text}")
            elif result['type'] == 'partial':
                sys.stdout.write(f"\r【部分结果】: {text:<40}")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n[主进程] 捕获到 Ctrl+C，正在优雅地关闭...")
    except Exception as e:
        print(f"[主进程] 发生错误: {e}")
    finally:
        # 发送结束信号给工作进程
        audio_queue.put(None)

        if 'stream' in locals() and stream.active:
            stream.stop()
            stream.close()
            print("[主进程] 音频流已关闭。")
        
        # 等待工作进程结束
        worker.join(timeout=5)
        if worker.is_alive():
            print("[主进程] 工作进程未在5秒内结束，强制终止。")
            worker.terminate()
        
        print("[主进程] 程序结束。")

def list_audio_devices():
    """列出音频设备的功能保持不变"""
    print(sd.query_devices())

if __name__ == "__main__":
    # 必须在 __name__ == "__main__" 保护下启动多进程
    multiprocessing.freeze_support() 
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'list':
        list_audio_devices()
    else:
        main()