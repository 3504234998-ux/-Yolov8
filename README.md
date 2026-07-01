# 基于改进 YOLOv8 的智能书籍检测系统

面向小样本场景的书籍目标检测算法改进项目。通过对 YOLOv8 损失函数与池化模块的多维度优化，在 350 张训练数据上实现 mAP50=0.925（+2%），并搭建 Gradio Web 检测系统。

## 改进方案

| 改进方向 | 模块 | mAP50 | Δ基线 |
|---------|------|:-----:|:-----:|
| 损失函数 | **SIoU** | 0.925 | **+1.9%** |
| 损失函数 | **DIoU** | 0.925 | **+1.9%** |
| 池化模块 | **SimSPPF** | 0.918 | **+1.2%** |
| 注意力 | C2f_SE | 0.903 | -0.3% |
| 基线 | YOLOv8n | 0.906 | — |

## 环境要求

- Python 3.8+
- PyTorch 2.4+
- CUDA (推荐) 或 CPU

```bash
conda create -n yolo python=3.8
conda activate yolo
pip install ultralytics torch gradio opencv-python pyyaml requests
```

## 项目结构

```
├── train.py                 # 训练入口 (改 EXPERIMENT_NAME/MODEL_CONFIG 切换实验)
├── evaluate_model.py        # 单模型测试集评估
├── evaluate_all_models.py   # 批量评估 + 生成对比表格
├── yolo_gui.py              # Gradio Web 检测系统
├── dataset/
│   ├── data.yaml            # 数据集配置 (path: dataset, nc: 1)
│   └── labels/              # YOLO 格式标签 (图片不上传)
├── ultralytics/
│   ├── nn/modules/block.py  # SimSPPF, SPPCSPC, C2f_SE, C2f_Light 等
│   ├── nn/modules/conv.py   # DWConv
│   ├── nn/tasks.py          # 模块注册 + init_criterion
│   ├── utils/loss.py        # BboxLoss (可配置 IoU)
│   ├── utils/metrics.py     # bbox_iou (SIoU 实现)
│   └── cfg/models/v8/       # 自定义模型 YAML 配置
├── thesis_experiment_reference.md  # 论文实验数据参考
├── project_experience.md           # 简历项目经历
└── evaluation_summary.txt          # 实验汇总
```

## 使用

### 训练

修改 `train.py` 顶部配置，然后：

```bash
python train.py
```

### 测试集评估

```bash
# 修改 evaluate_model.py 中的 EXP_NAME
python evaluate_model.py
```

### 启动 Web 检测系统

```bash
python yolo_gui.py
# 浏览器打开 http://127.0.0.1:7860
```

## 实验数据

共完成 16 组对比与消融实验，涵盖损失函数、池化模块、注意力机制、轻量化四个方向。详见 `evaluation_summary.txt` 及 `thesis_experiment_reference.md`。
