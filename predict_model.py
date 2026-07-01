from ultralytics import YOLO
import os
import cv2

# 预测函数
def predict_image(model, image_path):
    """对单个图像进行预测"""
    print(f"预测图像: {image_path}")
    
    # 进行预测
    results = model(image_path)
    
    # 显示结果
    results[0].show()
    
    # 保存结果
    save_dir = 'predictions'
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存预测结果图像
    result_path = os.path.join(save_dir, os.path.basename(image_path))
    results[0].save(result_path)
    print(f"预测结果已保存到: {result_path}")

# 批量预测函数
def batch_predict(model, image_dir):
    """批量预测目录中的所有图像"""
    print(f"批量预测目录: {image_dir}")
    
    # 获取目录中的所有图像文件
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(ext)])
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 批量预测
    results = model(image_files)
    
    # 保存结果
    save_dir = 'predictions'
    os.makedirs(save_dir, exist_ok=True)
    
    for i, result in enumerate(results):
        image_path = image_files[i]
        result_path = os.path.join(save_dir, os.path.basename(image_path))
        result.save(result_path)
        print(f"预测结果已保存到: {result_path}")

if __name__ == '__main__':
    # 加载最佳模型
    model = YOLO('runs/detect/yolov8n_book_detector/weights/best.pt')
    
    # 示例用法
    print("模型预测脚本")
    print("1. 预测单个图像")
    print("2. 批量预测目录中的图像")
    
    choice = input("请选择 (1/2): ")
    
    if choice == '1':
        image_path = input("请输入图像路径: ")
        if os.path.exists(image_path):
            predict_image(model, image_path)
        else:
            print("图像路径不存在！")
    elif choice == '2':
        image_dir = input("请输入图像目录路径: ")
        if os.path.exists(image_dir):
            batch_predict(model, image_dir)
        else:
            print("目录路径不存在！")
    else:
        print("无效选择！")