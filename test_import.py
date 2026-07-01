import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# 尝试导入Ultralytics
try:
    from ultralytics import YOLO

    print("Ultralytics imported successfully!")
    # 尝试加载模型
    model = YOLO("yolov8n.yaml")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error: {e}")
