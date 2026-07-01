"""
基于 Gradio 的智能书籍检测系统
支持: 图片上传 / 摄像头拍摄 / 模型切换 / 结果可视化.
"""

import os

import cv2

from ultralytics import YOLO

# ============================================================
# 模型配置
# ============================================================
MODELS = {
    "SIoU (最优 mAP50=0.925)": "runs/detect/improve_SIoU/weights/best.pt",
    "DIoU (mAP50=0.925)": "runs/detect/improve_DIoU/weights/best.pt",
    "SimSPPF (mAP50=0.918)": "runs/detect/improve_SimSPPF/weights/best.pt",
    "Baseline (mAP50=0.906)": "runs/detect/baseline_yolov8n/weights/best.pt",
    "C2fSE (mAP50=0.903)": "runs/detect/improve_C2fSE/weights/best.pt",
}

DEFAULT_MODEL = next(iter(MODELS.keys()))
CONF_THRESHOLD = 0.25


# ============================================================
# 核心检测函数
# ============================================================
def load_model(model_name):
    """加载选定的模型."""
    path = MODELS.get(model_name)
    if path and os.path.exists(path):
        return YOLO(path)
    # 回退到第一个存在的模型
    for name, p in MODELS.items():
        if os.path.exists(p):
            return YOLO(p)
    return None


def detect_and_draw(image, model):
    """执行检测并绘制结果."""
    results = model(image, conf=CONF_THRESHOLD)
    result = results[0]

    # 绘制边界框
    annotated = result.plot()

    # 统计信息
    num_books = len(result.boxes)
    if num_books > 0:
        confs = result.boxes.conf.cpu().numpy()
        info = (
            f"检测到 {num_books} 本书籍\n"
            f"平均置信度: {confs.mean():.3f}\n"
            f"最高置信度: {confs.max():.3f}\n"
            f"最低置信度: {confs.min():.3f}"
        )
    else:
        info = "未检测到书籍"

    return annotated, info


# ============================================================
# Gradio 回调
# ============================================================
def process_image(image, model_name, conf):
    """处理上传的图片."""
    global CONF_THRESHOLD
    CONF_THRESHOLD = conf

    model = load_model(model_name)
    if model is None:
        return None, "❌ 模型加载失败，请检查模型文件"

    annotated, info = detect_and_draw(image, model)
    return annotated, info


def process_video(video_path, model_name, conf):
    """处理视频文件."""
    global CONF_THRESHOLD
    CONF_THRESHOLD = conf

    model = load_model(model_name)
    if model is None:
        return None

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = "predictions/video_output.mp4"
    os.makedirs("predictions", exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_count = 0
    book_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        results = model(frame, conf=conf)
        annotated = results[0].plot()
        book_count += len(results[0].boxes)
        writer.write(annotated)

    cap.release()
    writer.release()

    avg_books = book_count / max(frame_count, 1)
    info = f"视频处理完成\n总帧数: {frame_count}\n检测书籍总数: {book_count}\n平均每帧: {avg_books:.1f} 本"
    return output_path, info


# ============================================================
# Gradio 界面
# ============================================================
def create_ui():
    """创建 Gradio 界面."""
    import gradio as gr

    css = """
    .container { max-width: 1100px; margin: auto; }
    .title { text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 10px; }
    .subtitle { text-align: center; color: #666; margin-bottom: 20px; }
    """

    with gr.Blocks(css=css, title="智能书籍检测系统") as demo:
        gr.HTML("""
        <div class="title">📚 基于改进 YOLOv8 的智能书籍检测系统</div>
        <div class="subtitle">融合 SIoU 损失函数与 SimSPPF 池化模块 | 小样本场景优化 | mAP50=0.925</div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                model_selector = gr.Dropdown(
                    choices=list(MODELS.keys()), value=DEFAULT_MODEL, label="🔧 检测模型", info="切换不同改进版本的模型"
                )
                conf_slider = gr.Slider(
                    minimum=0.05,
                    maximum=0.95,
                    value=0.25,
                    step=0.05,
                    label="🎯 置信度阈值",
                    info="降低阈值可检出更多书籍，但可能增加误检",
                )

                with gr.Accordion("📊 模型性能参考", open=False):
                    gr.Markdown("""
| 模型 | mAP50 | mAP50-95 | 特点 |
|------|:-----:|:--------:|------|
| SIoU | 0.925 | 0.761 | 最优，角度+形状惩罚 |
| DIoU | 0.925 | 0.767 | 中心距离惩罚，mAP50-95最高 |
| SimSPPF | 0.918 | 0.746 | 可学习池化，精度最高 |
| Baseline | 0.906 | 0.748 | YOLOv8n 原始基线 |
                    """)

            with gr.Column(scale=2):
                with gr.Tab("🖼️ 图片检测"):
                    image_input = gr.Image(type="numpy", label="上传图片", sources=["upload", "clipboard"])
                    img_btn = gr.Button("🔍 开始检测", variant="primary", size="lg")
                    with gr.Row():
                        image_output = gr.Image(type="numpy", label="检测结果")
                    info_output = gr.Textbox(label="检测信息", lines=4)

                with gr.Tab("🎬 视频检测"):
                    video_input = gr.Video(label="上传视频")
                    vid_btn = gr.Button("🎬 开始分析", variant="primary", size="lg")
                    with gr.Row():
                        video_output = gr.Video(label="处理结果")
                    vid_info = gr.Textbox(label="统计信息", lines=4)

        # 绑定事件
        img_btn.click(
            fn=process_image,
            inputs=[image_input, model_selector, conf_slider],
            outputs=[image_output, info_output],
        )
        vid_btn.click(
            fn=process_video,
            inputs=[video_input, model_selector, conf_slider],
            outputs=[video_output, vid_info],
        )

        gr.Markdown("""
---
**实验环境**: NVIDIA RTX 3050 Laptop (4GB) | PyTorch 2.4 | Ultralytics 8.4
**训练数据**: COCO 子集 350 张 | 50 epochs | 单类别 (book)
        """)

    return demo


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    # 检查关键模型是否存在
    missing = [n for n, p in MODELS.items() if not os.path.exists(p)]
    if missing:
        print("⚠️ 以下模型文件不存在:")
        for m in missing:
            print(f"  - {m}: {MODELS[m]}")
        print("请先运行 train.py 训练对应模型\n")

    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
