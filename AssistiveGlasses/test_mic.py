# check_mic.py
import sounddevice as sd

print("=========================================")
print("查询到的所有音频设备:")
print(sd.query_devices())
print("=========================================")