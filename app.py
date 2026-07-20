"""MedDenoise-Agent Web界面 (Gradio).

运行: python app.py  然后浏览器访问 http://localhost:7860
"""

import json

import gradio as gr
import numpy as np
from skimage import transform
from skimage.util import img_as_float

from meddenoise import data as med_data
from meddenoise.agent.coordinator import Coordinator
from meddenoise.report import generate_report

coordinator = Coordinator(verbose=False)


def _prepare(image: np.ndarray) -> np.ndarray:
    img = img_as_float(image)
    if img.ndim == 3:
        img = img.mean(axis=2)
    if max(img.shape) > 512:
        scale = 512 / max(img.shape)
        img = transform.resize(img, (int(img.shape[0] * scale),
                                     int(img.shape[1] * scale)))
    return np.clip(img, 0, 1)


def run_agent(image, add_noise, sigma):
    if image is None:
        return None, "请先上传图像", ""
    img = _prepare(image)
    reference = None
    if add_noise:
        reference = img
        img = med_data.add_gaussian_noise(img, sigma=sigma)
    result = coordinator.process(img, reference, "upload.png")
    report = generate_report(result, "upload.png")
    trace = json.dumps(result["trace"], ensure_ascii=False, indent=2, default=str)
    return (result["output"] * 255).astype(np.uint8), report, trace


def run_demo(sigma):
    reference = med_data.load_sample_images()["ct_phantom"]
    noisy = med_data.add_gaussian_noise(reference, sigma=sigma)
    result = coordinator.process(noisy, reference, "ct_phantom.png")
    report = generate_report(result, "ct_phantom.png")
    return ((noisy * 255).astype(np.uint8),
            (result["output"] * 255).astype(np.uint8), report)


with gr.Blocks(title="MedDenoise-Agent 医学影像智能去噪系统") as demo:
    gr.Markdown("# 🏥 MedDenoise-Agent 基于智能Agent的医学影像智能去噪与质量增强系统")
    gr.Markdown("感知Agent → 决策Agent → 执行Agent → 评估Agent 闭环自主处理")

    with gr.Tab("上传图像处理"):
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="输入医学影像", type="numpy")
                add_noise = gr.Checkbox(label="添加合成噪声(用于算法验证, 可计算PSNR/SSIM)", value=False)
                sigma_slider = gr.Slider(5, 50, value=25, label="合成噪声强度 σ")
                run_button = gr.Button("🤖 启动Agent处理", variant="primary")
            with gr.Column():
                output_image = gr.Image(label="处理结果")
        report_md = gr.Markdown(label="处理报告")
        trace_json = gr.Code(label="Agent决策轨迹", language="json")
        run_button.click(run_agent, [input_image, add_noise, sigma_slider],
                         [output_image, report_md, trace_json])

    with gr.Tab("CT体模演示"):
        demo_sigma = gr.Slider(5, 50, value=25, label="噪声强度 σ")
        demo_button = gr.Button("运行演示", variant="primary")
        with gr.Row():
            demo_noisy = gr.Image(label="噪声图像")
            demo_output = gr.Image(label="Agent处理结果")
        demo_report = gr.Markdown()
        demo_button.click(run_demo, [demo_sigma],
                          [demo_noisy, demo_output, demo_report])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
