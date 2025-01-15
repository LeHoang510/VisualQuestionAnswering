import torch
import numpy as np

def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available!")
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")
        for i in range(device_count):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device: {current_device} - {torch.cuda.get_device_name(current_device)}")
    else:
        print("CUDA is not available.")

import sys

def list_importable_directories():
    print("Directories Python can import from:")
    for directory in sys.path:
        print(directory)

if __name__ == "__main__":
    check_cuda()
    list_importable_directories()
