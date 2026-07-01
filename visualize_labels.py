import glob
import os

import cv2

# 输入和输出目录
input_dir = "dataset/images/train/"
output_dir = "dataset/visualized"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 读取类别名称
class_names = []
try:
    with open("dataset/category.names", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    print("Warning: category.names file not found, using default class name 'book'")
    class_names = ["book"]

# 获取所有jpg文件
image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
print(f"Found {len(image_files)} images to visualize")

# 处理每个图像
for img_path in image_files:
    # 获取对应的标签文件路径
    txt_path = os.path.splitext(img_path)[0] + ".txt"

    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image: {img_path}")
        continue

    height, width = img.shape[:2]

    # 读取并绘制标签
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 解析YOLO格式标签: class_id center_x center_y width height
                parts = line.split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                center_x = float(parts[1]) * width
                center_y = float(parts[2]) * height
                bbox_width = float(parts[3]) * width
                bbox_height = float(parts[4]) * height

                # 计算边界框坐标
                x1 = int(center_x - bbox_width / 2)
                y1 = int(center_y - bbox_height / 2)
                x2 = int(center_x + bbox_width / 2)
                y2 = int(center_y + bbox_height / 2)

                # 确保坐标在图像范围内
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width - 1, x2)
                y2 = min(height - 1, y2)

                # 获取类别名称
                class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

                # 绘制边界框
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 绘制类别名称
                cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 保存可视化结果
    output_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(output_path, img)
    print(f"Saved visualized image: {output_path}")

print(f"Visualization completed. {len(image_files)} images processed.")
