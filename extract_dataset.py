import os
import shutil
import glob

# 输入目录
ss_dir = 'dataset/ss'
train_dir = 'dataset/images/train'

# 输出目录
output_dir = 'dataset/ss_with_labels'

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 获取ss目录中的所有jpg文件
ss_images = glob.glob(os.path.join(ss_dir, '*.jpg'))
print(f"Found {len(ss_images)} images in ss directory")

# 处理每个图像
matched_count = 0
for img_path in ss_images:
    # 获取图像文件名
    img_name = os.path.basename(img_path)
    # 生成对应的txt文件名
    txt_name = os.path.splitext(img_name)[0] + '.txt'
    txt_path = os.path.join(train_dir, txt_name)
    
    # 检查txt文件是否存在
    if os.path.exists(txt_path):
        # 复制jpg文件
        dst_img_path = os.path.join(output_dir, img_name)
        shutil.copy(img_path, dst_img_path)
        
        # 复制txt文件
        dst_txt_path = os.path.join(output_dir, txt_name)
        shutil.copy(txt_path, dst_txt_path)
        
        matched_count += 1
        print(f"Copied: {img_name} and {txt_name}")
    else:
        print(f"Warning: No corresponding txt file found for {img_name}")

print(f"\nExtraction completed.")
print(f"Found {len(ss_images)} images in ss directory")
print(f"Matched {matched_count} image-label pairs")
print(f"Saved to: {output_dir}")