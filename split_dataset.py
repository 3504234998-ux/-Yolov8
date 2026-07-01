import os
import shutil
import random
from pathlib import Path

# 数据集根目录
DATASET_ROOT = Path('dataset')
SOURCE_IMAGES = DATASET_ROOT / 'images' / 'train'
SOURCE_LABELS = DATASET_ROOT / 'labels' / 'train'

# 目标目录
OUTPUT_DIR = DATASET_ROOT
TRAIN_IMG = OUTPUT_DIR / 'images' / 'train'
TRAIN_LBL = OUTPUT_DIR / 'labels' / 'train'
VAL_IMG = OUTPUT_DIR / 'images' / 'val'
VAL_LBL = OUTPUT_DIR / 'labels' / 'val'
TEST_IMG = OUTPUT_DIR / 'images' / 'test'
TEST_LBL = OUTPUT_DIR / 'labels' / 'test'

# 创建输出目录
for dir_path in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL, TEST_IMG, TEST_LBL]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 获取所有图像文件
image_files = list(SOURCE_IMAGES.glob('*.jpg'))
random.shuffle(image_files)

# 计算划分数量
total = len(image_files)
train_count = int(total * 0.7)
val_count = int(total * 0.2)
test_count = total - train_count - val_count

# 划分数据集
train_files = image_files[:train_count]
val_files = image_files[train_count:train_count+val_count]
test_files = image_files[train_count+val_count:]

# 复制文件到对应目录
def copy_files(files, img_dest, lbl_dest):
    for img_file in files:
        # 复制图像文件
        shutil.copy(img_file, img_dest / img_file.name)
        # 复制对应的标注文件
        txt_file = SOURCE_LABELS / (img_file.stem + '.txt')
        if txt_file.exists():
            shutil.copy(txt_file, lbl_dest / txt_file.name)

print(f"Total files: {total}")
print(f"Train files: {train_count}")
print(f"Val files: {val_count}")
print(f"Test files: {test_count}")

copy_files(train_files, TRAIN_IMG, TRAIN_LBL)
copy_files(val_files, VAL_IMG, VAL_LBL)
copy_files(test_files, TEST_IMG, TEST_LBL)

print("Dataset split completed successfully!")