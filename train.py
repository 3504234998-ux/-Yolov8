import os
# 解决OpenMP冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
import torch


# ============================================================
# 🔧 实验配置 —— 每次只改这两行即可切换实验
# ============================================================
EXPERIMENT_NAME = "compare_GhostConv"   # 实验名称 → 结果存入 runs/detect/<name>/
MODEL_CONFIG = "yolov8-ghost.yaml"             # 模型配置 (yaml=从零训练 / pt=预训练fine-tune)
IOU_TYPE = "CIoU"                       # IoU类型: CIoU / DIoU / GIoU / SIoU

# ============================================================
# 📋 实验清单 (复制粘贴到上面两行切换)
# ============================================================
# 基线:
#   MODEL_CONFIG = "yolov8n.pt"           EXPERIMENT_NAME = "baseline_yolov8n"
#
# 单模块改进:
#   MODEL_CONFIG = "yolov8-SimSPPF.yaml"  EXPERIMENT_NAME = "improve_SimSPPF"
#   MODEL_CONFIG = "yolov8-SPPCSPC.yaml"  EXPERIMENT_NAME = "improve_SPPCSPC"
#   MODEL_CONFIG = "yolov8n.pt"           EXPERIMENT_NAME = "improve_SIoU"        IOU_TYPE = "SIoU"
#   MODEL_CONFIG = "yolov8n.pt"           EXPERIMENT_NAME = "improve_DIoU"        IOU_TYPE = "DIoU"
#   MODEL_CONFIG = "yolov8n.pt"           EXPERIMENT_NAME = "improve_GIoU"        IOU_TYPE = "GIoU"
#   MODEL_CONFIG = "yolov8-SPPF-C2fSE.yaml"  EXPERIMENT_NAME = "improve_C2fSE"
#   MODEL_CONFIG = "yolov8-SPPF-C2fPSA.yaml" EXPERIMENT_NAME = "improve_C2fPSA"
#   MODEL_CONFIG = "yolov8-lightweight.yaml"  EXPERIMENT_NAME = "improve_DWConv"
#
# 消融组合:
#   MODEL_CONFIG = "yolov8-SPPCSPC.yaml"  EXPERIMENT_NAME = "combo_SPPCSPC_SIoU"  IOU_TYPE = "SIoU"
#   MODEL_CONFIG = "yolov8-SPPCSPC-C2fSE.yaml"  EXPERIMENT_NAME = "combo_SPPCSPC_C2fSE"
#   MODEL_CONFIG = "yolov8-SPPCSPC-SIoU-C2fSE.yaml" EXPERIMENT_NAME = "combo_all"  IOU_TYPE = "SIoU"
#
# 对比实验:
#   MODEL_CONFIG = "yolov8 - spp.yaml"    EXPERIMENT_NAME = "compare_SPP"
#   MODEL_CONFIG = "yolov8-SPPF.yaml"     EXPERIMENT_NAME = "compare_SPPF"
#   MODEL_CONFIG = "yolov8-ghost.yaml"    EXPERIMENT_NAME = "compare_GhostConv"


if __name__ == '__main__':
    # 检查GPU是否可用
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 自动选择设备
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    # 加载模型
    print(f"\n{'='*60}")
    print(f"实验: {EXPERIMENT_NAME}")
    print(f"模型: {MODEL_CONFIG}")
    print(f"IoU类型: {IOU_TYPE}")
    print(f"{'='*60}\n")

    model = YOLO(MODEL_CONFIG)

    # ============================================================
    # 关键修复1: yaml 配置需要迁移 yolov8n.pt 预训练权重
    # 否则等于从零训练，mAP50 只有 0.12
    # ============================================================
    if MODEL_CONFIG.endswith('.yaml') and os.path.exists('yolov8n.pt'):
        model = model.load('yolov8n.pt')
        print(f"✅ 已将 yolov8n.pt 预训练权重迁移到 {MODEL_CONFIG}")

    # ============================================================
    # 关键修复2: iou_type 必须设在 model.model (DetectionModel) 上
    # v8DetectionLoss 读的是 DetectionModel，不是 YOLO 包装器
    # ============================================================
    # ============================================================
    # 设置 IoU 类型 — 通过全局变量传递给 loss 模块 (绕过模型属性丢失问题)
    # ============================================================
    import ultralytics.utils.loss as loss_mod
    loss_mod.IOU_TYPE_OVERRIDE = IOU_TYPE if IOU_TYPE != "CIoU" else None
    print(f"✅ IoU 类型设置为: {IOU_TYPE}")

    # 训练参数
    train_params = {
        'data': 'dataset/data.yaml',
        'epochs': 50,                   # 训练轮数
        'batch': 4,                      # 批次大小 (适应4GB GPU)
        'imgsz': 640,                    # 图像尺寸
        'device': device,                # 自动选择设备
        'workers': 2 if torch.cuda.is_available() else 0,
        'name': EXPERIMENT_NAME,         # 实验名称 → 结果目录名
        'exist_ok': False,               # False=不覆盖已有实验, 避免误操作

        # 优化器
        'lr0': 0.01,                     # 初始学习率 (fine-tune 可提高到 0.01)
        'lrf': 0.01,                     # 最终学习率因子
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'cos_lr': True,

        # 数据增强
        'augment': True,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.1,
        'shear': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 0.5,                   # 50%概率马赛克增强
        'mixup': 0.1,                    # 少量mixup
        'copy_paste': 0.0,
        'multi_scale': False,
        'close_mosaic': 5,              # 最后5个epoch关闭马赛克

        # 验证
        'val': True,
        'plots': True,
    }

    print(f"\n训练参数: epochs={train_params['epochs']}, batch={train_params['batch']}, "
          f"lr0={train_params['lr0']}, iou_type={IOU_TYPE}")

    # 开始训练
    results = model.train(**train_params)

    print(f"\n✅ 训练完成！")
    print(f"结果保存路径: {results.save_dir}")
    print(f"最佳模型: {results.save_dir}/weights/best.pt")
