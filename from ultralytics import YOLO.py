from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolov8n.pt")

# 预测
results = model("image.jpg")

# 训练
model.train(data="coco128.yaml", epochs=100, imgsz=640)
