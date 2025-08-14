import torch

# 打印PyTorch版本
print(f"PyTorch version: {torch.__version__}")

# 检查CUDA是否可用 (这是最关键的一步)
is_available = torch.cuda.is_available()
print(f"CUDA available: {is_available}")

if is_available:
    # 如果可用，打印更多GPU信息
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU index: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
else:
    # 如果不可用，给出提示
    print("!!! CUDA is NOT available. PyTorch is running in CPU-only mode.")