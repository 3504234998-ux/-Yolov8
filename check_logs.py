import os

# 训练日志路径
log_path = os.path.join("runs", "detect", "yolov8n_book_detector", "train.log")

if os.path.exists(log_path):
    print(f"读取训练日志: {log_path}")
    print("=" * 80)
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        # 显示最后300行
        for line in lines[-300:]:
            print(line.rstrip())
else:
    print("训练日志文件不存在")
    # 检查是否有其他训练结果目录
    runs_dir = os.path.join("runs", "detect")
    if os.path.exists(runs_dir):
        print("可用的训练结果目录:")
        for item in os.listdir(runs_dir):
            item_path = os.path.join(runs_dir, item)
            if os.path.isdir(item_path):
                print(f"- {item}")
