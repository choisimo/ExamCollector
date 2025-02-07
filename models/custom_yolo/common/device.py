import psutil
import torch


class DeviceChecker:
    def __init__(self):
        pass

    def validate_training_device(self, device):
        if device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError(
                    "CUDA selected but not available!\n"
                    "1. Check NVIDIA drivers\n"
                    "2. Verify PyTorch CUDA version\n"
                    "3. Consider using CPU if no GPU available"
                )

        elif device == "hip":
            if not hasattr(torch.backends, 'rocmm') or not torch.backends.rocmm.is_available():
                raise ValueError(
                    "ROCm selected but not available!\n"
                    "1. Check AMD GPU drivers\n"
                    "2. Verify ROCm installation\n"
                    "3. Consider using CPU if no AMD GPU"
                )

        return True
