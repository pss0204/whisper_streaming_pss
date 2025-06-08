# test_cudnn.py
import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Current CUDA device name:", torch.cuda.get_device_name(0))
    # Output cuDNN version
    print("cuDNN version:", torch.backends.cudnn.version())
