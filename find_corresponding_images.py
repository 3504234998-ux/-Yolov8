import os
import shutil
import glob

# 输入目录
yolo_train_dir = 'dataset/images/train'
original_train_dir = 'dataset/images/train'

# 输出目录
output_dir = 'dataset/corresponding_images'

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 获取images/train目录中的所有jpg文件
yolo_images = glob.glob(os.path.join(yolo_train_dir, '*.jpg'))
print(f"Found {len(yolo_images)} images in images/train directory")

# 处理每个图像
found_count = 0
not_found_count = 0

for img_path in yolo_images:
    # 获取图像文件名
    img_name = os.path.basename(img_path)
    # 构建原始train目录中的文件路径
    original_img_path = os.path.join(original_train_dir, img_name)
    
    # 检查文件是否存在
    if os.path.exists(original_img_path):
        # 复制文件到输出目录
        dst_path = os.path.join(output_dir, img_name)
        shutil.copy(original_img_path, dst_path)
        found_count += 1
        print(f"Found and copied: {img_name}")
    else:
        not_found_count += 1
        print(f"Not found: {img_name}")

print(f"\nSearch completed.")
print(f"Total images in images/train: {len(yolo_images)}")
print(f"Found and copied: {found_count}")
print(f"Not found: {not_found_count}")
print(f"Saved to: {output_dir}")