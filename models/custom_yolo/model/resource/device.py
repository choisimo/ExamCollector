import psutil
import torch
class DeviceChecker:

    def check_device_validity(self, selected_device):
        if selected_device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA를 선택했지만 시스템에서 CUDA가 지원되지 않습니다. "
                                 "NVIDIA GPU 드라이버 및 PyTorch 설치를 확인해주세요.")

            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                raise ValueError("사용 가능한 NVIDIA GPU가 없습니다.")

        elif selected_device == "hip":
            if not torch.backends.rocmm.is_available():
                raise ValueError("ROCm를 선택했지만 시스템에서 ROCm가 지원되지 않습니다. "
                                 "AMD GPU 드라이버 및 ROCm 설치를 확인해주세요.")

            # AMD GPU 개수 확인
            amd_gpus = [gpu for gpu in psutil.cpu_percent() if 'AMD' in gpu]
            if not amd_gpus:
                raise ValueError("사용 가능한 AMD GPU가 없습니다.")

        return True

    def get_available_device(self):
        try:
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                print(f"사용 가능한 NVIDIA GPU가 {num_gpus}개 있습니다.")
                return "cuda"

            elif torch.backends.rocmm.is_available():
                amd_gpus = [gpu for gpu in psutil.cpu_percent() if 'AMD' in gpu]
                if amd_gpus:
                    print("사용 가능한 AMD GPU를 감지했습니다.")
                    return "hip"

        except Exception as e:
            print(f"디바이스 검색 중 오류가 발생했습니다: {str(e)}")

        # 모든 GPU 사용 불가능 시 CPU로 fallback
        print("GPU 사용을 지원하지 않습니다. CPU로 전환합니다.")
        return "cpu"
