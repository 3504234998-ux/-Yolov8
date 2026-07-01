from __future__ import annotations

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv

from ultralytics import YOLO

# ============================================================
# 模型对比评估脚本 — 生成论文第5章实验表格数据
# ============================================================
# 用法: 跑完所有实验后, 运行此脚本
#   python evaluate_all_models.py
# 输出: model_comparison.csv (可导入Excel做表格)

EXPERIMENTS = [
    # === 基线 ===
    ("baseline_yolov8n", "YOLOv8n (基线)"),
    # === 单模块改进: 空间金字塔池化 ===
    ("improve_SPP", "SPP"),
    ("improve_SPPF", "SPPF"),
    ("improve_SimSPPF", "SimSPPF (本文)"),
    ("improve_SPPCSPC", "SPPCSPC (本文)"),
    # === 单模块改进: 损失函数 ===
    ("improve_SIoU", "SIoU (本文)"),
    ("improve_DIoU", "DIoU"),
    ("improve_GIoU", "GIoU"),
    # === 单模块改进: 注意力机制 ===
    ("improve_C2fSE", "C2f_SE (本文)"),
    ("improve_C2fPSA", "C2f_PSA"),
    # === 单模块改进: 轻量化 ===
    ("improve_DWConv", "C2f_DWConv (本文)"),
    ("compare_GhostConv", "GhostConv"),
    # === 消融组合 ===
    ("combo_SPPCSPC_SIoU", "SPPCSPC + SIoU"),
    ("combo_SPPCSPC_C2fSE", "SPPCSPC + C2f_SE"),
    ("combo_all", "SPPCSPC + SIoU + C2f_SE (本文最终)"),
]

BASE_DIR = "runs/detect"
DATA_YAML = "dataset/data.yaml"


def evaluate_model(exp_dir: str, exp_name: str) -> dict | None:
    """评估单个模型并返回指标字典."""
    best_pt = os.path.join(BASE_DIR, exp_dir, "weights", "best.pt")
    if not os.path.exists(best_pt):
        print(f"  ⚠️ 跳过 {exp_name}: 模型文件不存在 ({best_pt})")
        return None

    try:
        model = YOLO(best_pt)
        results = model.val(data=DATA_YAML, split="test", verbose=False)

        # 估计参数量 (从模型文件大小粗略估计)
        import torch

        ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
        if "model" in ckpt and hasattr(ckpt["model"], "yaml"):
            ckpt["model"].yaml
            # 从yaml中获取nc信息
        params = sum(p.numel() for p in model.model.parameters()) / 1e6  # M

        return {
            "name": exp_name,
            "mAP50": round(results.box.map50, 4),
            "mAP50_95": round(results.box.map, 4),
            "Precision": round(results.box.mp, 4),
            "Recall": round(results.box.mr, 4),
            "Params_M": round(params, 2),
        }
    except Exception as e:
        print(f"  ❌ 评估失败 {exp_name}: {e}")
        return None


def main():
    print("=" * 70)
    print("📊 模型对比评估 — 论文实验表格")
    print("=" * 70)

    results_list = []
    for exp_dir, exp_name in EXPERIMENTS:
        print(f"\n评估: {exp_name} ({exp_dir})")
        result = evaluate_model(exp_dir, exp_name)
        if result:
            results_list.append(result)
            print(
                f"  mAP50={result['mAP50']:.4f}  "
                f"mAP50-95={result['mAP50_95']:.4f}  "
                f"P={result['Precision']:.4f}  R={result['Recall']:.4f}  "
                f"Params={result['Params_M']:.2f}M"
            )

    if not results_list:
        print("\n⚠️ 没有找到任何已完成的实验。请先运行 train.py。")
        return

    # 保存 CSV
    csv_path = "model_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "mAP50", "mAP50_95", "Precision", "Recall", "Params_M"])
        writer.writeheader()
        writer.writerows(results_list)

    print(f"\n{'=' * 70}")
    print(f"📋 结果已保存到: {csv_path}")
    print(f"{'=' * 70}")

    # 打印对比表格
    print(f"\n{'实验名称':<35} {'mAP50':>8} {'mAP50-95':>10} {'P':>8} {'R':>8} {'Params(M)':>10}")
    print("-" * 80)
    for r in results_list:
        print(
            f"{r['name']:<35} {r['mAP50']:>8.4f} {r['mAP50_95']:>10.4f} "
            f"{r['Precision']:>8.4f} {r['Recall']:>8.4f} {r['Params_M']:>10.2f}"
        )


if __name__ == "__main__":
    main()
