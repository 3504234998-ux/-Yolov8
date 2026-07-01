# YOLOv8 改进论文 — 实验数据与方案参考

> 以下数据基于真实实验，可直接用于论文第3/4/5章撰写。

---

## 一、项目基本信息

| 项目 | 内容 |
|------|------|
| 任务 | 书籍目标检测（单类别） |
| 基线模型 | YOLOv8n (yolov8n.pt 预训练权重) |
| 数据集 | COCO 子集，350 张训练 / 175 张验证 / 88 张测试 |
| 输入尺寸 | 640×640 |
| 训练轮数 | 50 epochs |
| 批次大小 | 4 |
| 优化器 | AdamW (自动选择), lr=0.002, cos_lr 衰减 |
| 硬件 | NVIDIA RTX 3050 Laptop GPU (4GB) |
| 框架 | PyTorch 2.4.1 + Ultralytics 8.4.19 |

---

## 二、基线算法 YOLOv8 简介（用于 3.1 节）

YOLOv8 由 Ultralytics 提出，是一种单阶段目标检测器，由三个主要部分组成：

**Backbone（特征提取网络）**：
- 采用 CSPDarknet 结构，核心模块为 C2f（CSP Bottleneck with 2 convolutions）
- C2f 通过 split-transform-merge 策略减少计算量，同时保持梯度流
- 使用 SPPF（Spatial Pyramid Pooling - Fast）实现多尺度特征融合
- 5 次 3×3 stride=2 卷积下采样，输出 P3/8、P4/16、P5/32 三个尺度特征图

**Neck（特征融合网络）**：
- 采用 FPN（Feature Pyramid Network）+ PAN（Path Aggregation Network）结构
- 自顶向下传递语义信息，自底向上传递位置信息
- 使用 Concat 操作融合多尺度特征

**Head（检测头）**：
- 解耦检测头，分别预测分类和回归
- 采用 Anchor-Free 方式，直接在特征图上回归边界框
- 使用 DFL（Distribution Focal Loss）+ CIoU Loss 进行边界框回归
- 使用 BCE Loss 进行目标分类

**YOLOv8n 参数量**: 3,011,043 参数, 8.2 GFLOPs

---

## 三、完整实验数据（用于 5.3 节实验表格）

### 3.1 单模块改进实验

| 实验名称 | mAP50 | mAP50-95 | Precision | Recall | Δ mAP50 |
|----------|:-----:|:--------:|:---------:|:------:|:-------:|
| baseline_yolov8n（基线） | 0.9061 | 0.7478 | 0.9277 | 0.8138 | — |
| **improve_SIoU**（SIoU损失） | **0.9250** | 0.7606 | 0.9444 | 0.8125 | **+1.89%** |
| **improve_DIoU**（DIoU损失） | 0.9247 | **0.7665** | 0.9422 | 0.8085 | **+1.86%** |
| **improve_SimSPPF**（简化SPPF） | 0.9182 | 0.7459 | **0.9512** | 0.7872 | **+1.21%** |
| improve_C2fSE（SE注意力） | 0.9027 | 0.7229 | 0.8997 | **0.8191** | -0.34% |
| improve_GIoU（GIoU损失） | 0.8974 | 0.7394 | 0.8978 | 0.8032 | -0.87% |
| improve_SPPCSPC（SPPCSPC） | 0.8599 | 0.6736 | 0.9084 | 0.6857 | -4.62% |
| improve_C2fPSA（PSA注意力） | 0.7940 | 0.6045 | 0.8116 | 0.6702 | -11.21% |
| improve_DWConv（深度可分离卷积） | 0.3815 | 0.2289 | 0.5065 | 0.4043 | -52.46% |

### 3.2 消融组合实验

| 实验名称 | mAP50 | mAP50-95 | Precision | Recall | Δ mAP50 |
|----------|:-----:|:--------:|:---------:|:------:|:-------:|
| combo_SPPCSPC_C2fSE | 0.8850 | 0.7159 | 0.9530 | 0.7287 | -2.11% |
| combo_SPPCSPC_SIoU | 0.8716 | 0.6893 | 0.8395 | 0.7660 | -3.45% |
| combo_all（SPPCSPC+SIoU+C2fSE） | 0.8428 | 0.6555 | 0.8471 | 0.7181 | -6.33% |

### 3.3 对比实验

| 实验名称 | mAP50 | mAP50-95 | Precision | Recall | Δ mAP50 |
|----------|:-----:|:--------:|:---------:|:------:|:-------:|
| compare_SPP（SPP池化） | 0.9166 | 0.7434 | 0.8710 | **0.8620** | +1.05% |
| compare_SPPF（SPPF池化） | 0.9061 | 0.7478 | 0.9277 | 0.8138 | 0.00% |
| compare_GhostConv | 0.1171 | 0.0552 | 0.2442 | 0.1489 | -78.89% |

---

## 四、各改进模块详细说明（用于 3.3 节）

### 4.1 SimSPPF（简化空间金字塔池化）

**改进动机**：标准 SPPF 使用 3 次串行 MaxPool 进行多尺度特征融合。MaxPool 操作不可导，可能造成梯度流动不充分。SimSPPF 将 MaxPool 替换为 Conv+ReLU，保持感受野等效的同时改善梯度传播。

**改动内容**：
- 将 SPPF 模块中的 `nn.MaxPool2d(kernel_size=5)` 替换为 3 个串行的 `Conv(c, c, 5, 1)` + ReLU 激活
- 保持与 SPPF 等效的感受野（3 次 5×5 Conv = 1 次 5×5 + 1 次 9×9 + 1 次 13×13 感受野）
- 其他 Backbone 结构不变

**实验结果**：mAP50 +1.21%，Precision +2.4%（0.9277→0.9512）

### 4.2 SIoU（Scylla-IoU 边界框损失）

**改进动机**：基线使用的 CIoU Loss 仅考虑重叠面积、中心点距离和长宽比三个几何因素。SIoU 额外引入了**角度惩罚项**和**形状惩罚项**，重新定义了距离度量的方式。

**SIoU 公式**（4 个组成部分）：
```
Λ = cos(2α - π/2)                              # 角度代价
γ = Λ - 2
ρ_x = (s_cx / c_w)²,  ρ_y = (s_cy / c_h)²       # 中心距离归一化
Δ = 2 - e^(-γ·ρ_x) - e^(-γ·ρ_y)                 # 距离代价
Ω = (1 - e^(-ω_w))⁴ + (1 - e^(-ω_h))⁴            # 形状代价
SIoU = IoU - 0.5×(Δ + Ω)                        # 最终损失
```

**改动内容**：
- 在 `bbox_iou()` 函数中新增 `SIoU=True` 分支
- 在 `BboxLoss` 和 `v8DetectionLoss` 中增加 `iou_type` 参数支持
- 通过全局变量 `IOU_TYPE_OVERRIDE` 实现训练时动态切换

**实验结果**：mAP50 +1.89%（最优），同时 mAP50-95 +1.3%

### 4.3 DIoU（Distance-IoU 边界框损失）

**改进动机**：CIoU 中的长宽比惩罚项 v 在某些情况下会导致训练不稳定。DIoU 直接使用中心点距离作为惩罚，公式更简洁，优化目标更明确。

**DIoU 公式**：
```
DIoU = IoU - ρ²(b, b_gt) / c²
```
其中 ρ²(b, b_gt) 是预测框和真实框中心点的欧氏距离，c 是最小外接矩形的对角线长度。

**实验结果**：mAP50 +1.86%，mAP50-95 +1.87%（在 mAP50-95 上优于 SIoU）

### 4.4 C2f_SE（通道注意力增强 C2f）

**改进动机**：标准 C2f 模块对所有通道一视同仁，但不同通道对检测任务的贡献不同。引入 SE（Squeeze-and-Excitation）注意力机制，让模型自动学习通道间的重要性权重。

**SE 注意力流程**：
```
AvgPool → FC → ReLU → FC → Sigmoid → Channel Weight × Feature
```

**改动内容**：
- 在 Bottleneck 的第二个 Conv 后插入 SEAttention 模块
- SE 使用 reduction=16，adaptive average pooling 压缩空间信息
- Backbone 中所有 C2f 替换为 C2f_SE

**实验结果**：mAP50 -0.34%，基本持平。SE 注意力在 350 张小数据集上未能体现优势，但提供了对比依据。

### 4.5 SPPCSPC（空间金字塔池化 + 跨阶段部分连接）

**改进动机**：借鉴 YOLOv7 的 SPPCSPC 设计，将 SPP 的多尺度池化与 CSP 结构结合，期望获得更强的多尺度特征融合能力。

**结构**：SPPCSPC 包含两条分支——主分支通过多尺度 MaxPool 提取特征，副分支直接传递原始特征，最后 Concat 融合。

**实验结果**：mAP50 -4.62%，在 350 张训练数据下性能下降明显。复杂结构在小数据集上容易过拟合或欠收敛。

### 4.6 C2f_Light / DWConv 轻量化

**改进动机**：用 Depthwise Separable Convolution 替换 Bottleneck 中的标准 3×3 Conv，期望在保持精度的同时减少参数量。

**结构差异**：
- C2f: `Conv(1×1) + Bottleneck(Conv(3×3) + Conv(3×3)) + Conv(1×1)`
- C2f_Light: `Conv(1×1) + Bottleneck(DWConv(3×3) + DWConv(3×3)) + Conv(1×1)`

**参数量对比**：C2f_Light 2.22M vs 标准 YOLOv8n 3.01M（-26.1%）

**实验结果**：mAP50 仅 0.3815（-52.5%）。轻量化模块在小数据集上难以收敛。

---

## 五、损失函数切换技术实现（用于 4.2 节）

### 伪代码

```
// train.py
IOU_TYPE = "SIoU"  // 可选: CIoU, DIoU, GIoU, SIoU

// 设置全局 IoU 覆盖变量
loss_mod.IOU_TYPE_OVERRIDE = IOU_TYPE

// 训练: model.train()

// loss.py — v8DetectionLoss.__init__()
if IOU_TYPE_OVERRIDE is not None:
    self.iou_type = IOU_TYPE_OVERRIDE
elif iou_type is not None:
    self.iou_type = iou_type
else:
    self.iou_type = "CIoU"

// loss.py — BboxLoss.forward()
iou = bbox_iou(pred_bbox, target_bbox, **self.iou_kwargs)
loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
```

---

## 六、实验结果分析要点（用于 5.4 节）

### 6.1 有效改进

1. **损失函数改进（SIoU/DIoU）效果最佳**：零参数成本的改进，mAP50 提升 1.9%，证实了角度和距离惩罚项在边界框回归中的有效性。

2. **SimSPPF 优于 SPPF**：将 MaxPool 替换为可学习的 Conv+ReLU 操作，在小数据集上体现了更好的特征适应性。

3. **SPP 略优于 SPPF**：并行池化（SPP）比串行池化（SPPF）在 recall 上更高（0.862 vs 0.814），适合需要高召回的场景。

### 6.2 失败的改进及其原因

1. **SPPCSPC 在全实验中都呈副作用**：YOLOv7 级的复杂模块需要更多数据支撑，350 张图不足以发挥其多分支 CSP 结构的优势。

2. **组合改进不如单模块**：SPPCSPC + C2fSE + SIoU 三组合的 mAP50（0.843）低于任一单独模块，说明模块间存在负面交互效应。

3. **轻量化模块在小数据集上不可行**：GhostConv、C2f_DWConv 预训练权重无法迁移，小数据集从头训练无法收敛。

4. **C2fPSA 结构差异过大**：PSA 注意力与标准 Bottleneck 差异大，权重迁移不完整。

### 6.3 核心启示

- 在小数据集（<500张）上，**零参数改进**（损失函数、激活函数替换）优于**结构改进**（新模块、注意力机制）
- 模块组合时需注意**权重兼容性**和**模块间协同效应**
- **不要盲目追求复杂结构**，简单有效的改进才是工程实践的最佳选择

---

## 七、训练配置参考（用于 4.1 节）

```python
train_params = {
    'data': 'dataset/data.yaml',    # 数据集配置
    'epochs': 50,                    # 训练轮数
    'batch': 4,                     # 批大小（4GB 显存约束）
    'imgsz': 640,                   # 输入图像尺寸
    'lr0': 0.01,                    # 初始学习率（自动优化器会重设）
    'lrf': 0.01,                    # 最终学习率因子
    'cos_lr': True,                 # 余弦退火调度
    'warmup_epochs': 3,             # 预热轮数
    'weight_decay': 0.0005,         # 权重衰减
    'mosaic': 0.5,                  # 马赛克增强概率 50%
    'mixup': 0.1,                   # MixUp 增强概率 10%
    'close_mosaic': 5,              # 最后 5 epoch 关闭马赛克
    'hsv_h': 0.015,                 # HSV 色调增强
    'hsv_s': 0.7,                   # HSV 饱和度增强
    'hsv_v': 0.4,                   # HSV 亮度增强
    'fliplr': 0.5,                  # 水平翻转概率
}
```

---

## 八、论文章节撰写建议

| 章节 | 对应数据 | 核心要点 |
|------|---------|---------|
| 3.1 原算法介绍 | 第二部分 | YOLOv8 结构（C2f/SPPF/CIoU），参数量，FLOPS |
| 3.2 问题分析 | 第四、六部分 | SPPF 梯度问题、CIoU 角度缺失、通道无注意力 |
| 3.3 改进设计 | 第四部分 | SIoU/DIoU/SimSPPF 的设计思想 & 公式 |
| 3.4 章节小结 | 第六部分 | 损失函数改进行之有效，轻量化不适用 |
| 4.1 实现工具 | 第七部分 | PyTorch + CUDA + 硬件配置 |
| 4.2 模块实现 | 第五部分 | SIoU 伪代码、SimSPPF 设计、C2f_SE 结构 |
| 5.1 实验目标 | — | 验证损失函数、池化、注意力三个方向的改进效果 |
| 5.2 实验设计 | 第七部分 | 数据集、评价指标（mAP50/mAP50-95/P/R）、对比方案 |
| 5.3 实验结果 | 第三部分 | 将表格数据填入论文 |
| 5.4 结果分析 | 第六部分 | 逐条分析有效/失败的原因 |
| 5.5 章节小结 | 第六部分 6.3 | 核心启示 |
