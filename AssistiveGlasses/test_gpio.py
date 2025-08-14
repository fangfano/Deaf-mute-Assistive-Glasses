import subprocess
import time
import sys

# --- 配置参数 ---
# 您要监测的 wPi 引脚编号 (来自`gpio readall`的 wPi 列)
WPI_PIN_NUMBER = 5

# 刷新间隔（秒）
CHECK_INTERVAL = 1.0
# -----------------

def check_gpio_status():
    """
    通过调用外部`gpio`命令来获取引脚状态。
    """
    # 构建要执行的命令，使用列表形式更安全
    command = ["gpio", "read", str(WPI_PIN_NUMBER)]
    
    try:
        # 执行命令并捕获输出
        result = subprocess.run(
            command, 
            capture_output=True,  # 捕获标准输出和标准错误
            text=True,            # 将输出解码为文本
            check=True,           # 如果命令返回错误码，则抛出异常
            timeout=5             # 设置5秒超时
        )
        
        # `result.stdout` 的内容会是 "0\n" 或 "1\n"，.strip()可以去除末尾的换行符
        status = result.stdout.strip()
        return status

    except FileNotFoundError:
        print("\n[错误] 'gpio' 命令未找到。")
        print("请确认 WiringOP 已经正确安装并且在系统的 PATH 路径中。")
        sys.exit(1) # 退出程序
    except subprocess.CalledProcessError as e:
        print(f"\n[错误] 'gpio' 命令执行失败: {e}")
        print("请检查引脚编号是否正确，或尝试使用 `sudo` 运行此脚本。")
        sys.exit(1)
    except Exception as e:
        print(f"\n[未知错误] 发生错误: {e}")
        sys.exit(1)


# --- 主程序 ---
print("--- GPIO 状态监测器 (通过调用 'gpio' 命令) ---")
print(f"正在监测 wPi 引脚: {WPI_PIN_NUMBER}")
print("按下 Ctrl+C 即可退出程序。")
print("-" * 50)

try:
    while True:
        # 获取状态
        pin_value = check_gpio_status()
        
        # 根据状态值显示 HIGH/LOW
        display_text = "HIGH" if pin_value == "1" else "LOW"
        
        # 动态刷新，在同一行显示结果
        print(f"[状态] wPi 引脚 {WPI_PIN_NUMBER}: {display_text} (原始值: {pin_value})   ", end='\r')
        
        # 等待指定的间隔时间
        time.sleep(CHECK_INTERVAL)

except KeyboardInterrupt:
    # 用户按下 Ctrl+C 时，优雅地退出
    print("\n\n[INFO] 用户中断，程序结束。")
    sys.exit(0)