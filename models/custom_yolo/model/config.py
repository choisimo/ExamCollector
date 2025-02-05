# config.py

# 모델 관련 설정
DEFAULT_MODEL_PATH = 'yolov8n-seg.pt'
SAVE_FORMAT = 'onnx'

# 훈련 설정
TRAIN_DATA_YAML = './learning_model/custom_data.yaml'
IMAGE_SIZE = 640
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 16

# 저장 설정
SAVED_MODEL_DIR = './saved_models/'

# 디바이스 설정
DEVICE_CHOICES = {
    "cuda": "CUDA (NVIDIA GPU)",
    "hip": "ROCm (AMD GPU)",
    "cpu": "CPU"
}

# 메모리 모니터링 설정
MONITORING_INTERVAL_SEC = 1
