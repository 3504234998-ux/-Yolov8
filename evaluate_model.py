import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO

if __name__ == '__main__':
    # ============================================================
    # 🔧 改这里切换实验
    # ============================================================
    EXP_NAME = "compare_GhostConv"   # 实验名称
    MODEL_PATH = f"runs/detect/{EXP_NAME}/weights/best.pt"

    print(f"评估实验: {EXP_NAME}")
    print(f"模型路径: {MODEL_PATH}")

    # 加载最佳模型
    model = YOLO(MODEL_PATH)

    print("开始在测试集上评估模型...")

    # 在测试集上评估（结果输出到实验目录下，不污染 runs/detect/val/）
    results = model.val(
        data='dataset/data.yaml',
        split='test',
        project='runs/detect',
        name=f'{EXP_NAME}_eval',   # 存到 runs/detect/baseline_yolov8n_eval/
        exist_ok=True,
    )

    # 打印评估结果
    print("\n" + "=" * 50)
    print(f"实验: {EXP_NAME}")
    print(f"mAP50:     {results.box.map50:.4f}")
    print(f"mAP50-95:  {results.box.map:.4f}")
    print(f"精确率 (P): {results.box.mp:.4f}")
    print(f"召回率 (R): {results.box.mr:.4f}")
    print("=" * 50)

    # 追加到评估汇总文件
    summary_file = 'evaluation_summary.txt'
    if not os.path.exists(summary_file):
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"{'实验':<35} {'mAP50':>8} {'mAP50-95':>10} {'P':>8} {'R':>8}\n")
            f.write("-" * 70 + "\n")

    with open(summary_file, 'a', encoding='utf-8') as f:
        f.write(f"{EXP_NAME:<35} {results.box.map50:>8.4f} {results.box.map:>10.4f} "
                f"{results.box.mp:>8.4f} {results.box.mr:>8.4f}\n")

    print(f"\n结果已追加到: {summary_file}")
