"""MedDenoise-Agent 展示前端后端服务 (FastAPI).

运行: python server.py  然后浏览器访问 http://localhost:8000
"""

import base64
import csv
import io
import json
from pathlib import Path

import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()
from skimage import io as skio, transform
from skimage.util import img_as_float

from meddenoise import data as med_data
from meddenoise.agent import llm
from meddenoise.agent.ai_pipeline import run_ai_pipeline
from meddenoise.agent.chat import ChatAgent
from meddenoise.agent.coordinator import Coordinator
from meddenoise.tools import metrics
from meddenoise.tools.registry import ToolExecutor

ROOT = Path(__file__).resolve().parent
SET12_DIR = ROOT / "data" / "datasets" / "Set12"
BENCHMARK_CSV = ROOT / "experiments" / "results" / "benchmark.csv"

app = FastAPI(title="MedDenoise-Agent")
coordinator = Coordinator(verbose=False)
chat_agent = ChatAgent()


def _to_b64(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    skio.imsave(buffer, (np.clip(image, 0, 1) * 255).astype(np.uint8),
                format="png")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _prepare(image: np.ndarray) -> np.ndarray:
    img = img_as_float(image)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    if max(img.shape) > 512:
        scale = 512 / max(img.shape)
        img = transform.resize(img, (int(img.shape[0] * scale),
                                     int(img.shape[1] * scale)))
    return np.clip(img, 0, 1)


@app.get("/api/samples")
def list_samples():
    samples = [{"id": "ct_phantom", "name": "CT体模 (Shepp-Logan)"}]
    if SET12_DIR.exists():
        samples += [{"id": f"set12_{f.stem}", "name": f"Set12 #{f.stem}"}
                    for f in sorted(SET12_DIR.glob("*.png"))]
    return samples


def _load_sample(sample_id: str) -> np.ndarray:
    if sample_id == "ct_phantom":
        return med_data.load_sample_images()["ct_phantom"]
    if sample_id.startswith("set12_"):
        return med_data.load_image(str(SET12_DIR / f"{sample_id[6:]}.png"))
    raise ValueError(f"未知示例: {sample_id}")


@app.post("/api/process")
async def process(file: UploadFile | None = File(None),
                  sample_id: str = Form(""),
                  add_noise: bool = Form(True),
                  sigma: float = Form(25.0)):
    if file is not None and file.filename:
        raw = skio.imread(io.BytesIO(await file.read()))
        image = _prepare(raw)
        filename = file.filename
    elif sample_id:
        image = _load_sample(sample_id)
        filename = f"{sample_id}.png"
    else:
        return {"error": "请上传图像或选择示例"}

    reference = None
    if add_noise:
        reference = image
        image = med_data.add_gaussian_noise(image, sigma=sigma)

    result = coordinator.process(image, reference, filename)
    response = {
        "input_image": _to_b64(image),
        "output_image": _to_b64(result["output"]),
        "perception": result["perception"],
        "planner": result["planner"],
        "history": result["history"],
        "trace": [{"tool": t["tool"], "args": t["args"],
                   "result": {k: v for k, v in t["result"].items()}}
                  for t in result["trace"]],
        "evaluation": result["evaluation"],
    }
    if reference is not None:
        response["reference_image"] = _to_b64(reference)
        response["noisy_metrics"] = metrics.full_reference_metrics(reference, image)
    return response


@app.post("/api/ai_process")
async def ai_process(file: UploadFile | None = File(None),
                     sample_id: str = Form(""),
                     add_noise: bool = Form(True),
                     sigma: float = Form(25.0)):
    if file is not None and file.filename:
        raw = skio.imread(io.BytesIO(await file.read()))
        image = _prepare(raw)
    elif sample_id:
        image = _load_sample(sample_id)
    else:
        return {"error": "请上传图像或选择示例"}

    reference = None
    noisy = image
    if add_noise:
        reference = image
        noisy = med_data.add_gaussian_noise(image, sigma=sigma)

    def stream():
        for event in run_ai_pipeline(noisy, reference):
            if event["type"] == "result":
                payload = {
                    "type": "result",
                    "perception": event["perception"],
                    "plan": event["plan"],
                    "evaluation": event["evaluation"],
                    "history": [{"plan": h["plan"],
                                 "evaluation": h["evaluation"]}
                                for h in event["history"]],
                    "llm": event["llm"],
                    "input_image": _to_b64(noisy),
                    "output_image": _to_b64(event["output"]),
                }
                if reference is not None:
                    payload["reference_image"] = _to_b64(reference)
                    payload["noisy_metrics"] = metrics.full_reference_metrics(
                        reference, noisy)
                yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
            else:
                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


CHAT_SYSTEM = (
    "你是「MedDenoise-Agent」医学影像智能去噪系统的AI助手, 精通VT-BM3D论文"
    "(Peng et al., Signal Processing 243 (2026) 110417: 结构感知+噪声自适应联合优化, "
    "相对BM3D平均PSNR提升2.04dB, 复杂纹理最大4.19dB)、DnCNN深度学习去噪与传统方法。"
    "本系统: 多Agent架构(感知→AI调参→执行→反思), 支持Set12/BSD300数据集, "
    "本地实测σ=25时 DnCNN 29.58dB > BM3D 29.19 > VT-BM3D 29.03 (纯高斯噪声)。"
    "用简洁专业的中文回答, 可用Markdown, 专业名词(PSNR/SSIM等)保留英文。"
    "若用户想处理图像, 建议其直接说「用σ=25处理Set12第5张」这样的指令。"
)


@app.post("/api/chat")
async def chat(message: str = Form(...), history: str = Form("[]")):
    out = chat_agent.reply(message)
    action = out.get("action")
    if action is None and llm.is_llm_available():
        try:
            msgs = [m for m in json.loads(history) if m.get("content")][-8:]
        except json.JSONDecodeError:
            msgs = []
        msgs.append({"role": "user", "content": message})
        reply = llm.llm_chat(CHAT_SYSTEM, msgs)
        if reply:
            return {"reply": reply, "llm": True}
    response = {"reply": out["reply"]}
    if action and action["type"] == "process":
        image = _load_sample(action["sample_id"])
        noisy = med_data.add_gaussian_noise(image, sigma=action["sigma"])
        if action.get("method"):
            executor = ToolExecutor(noisy, image, action["sample_id"])
            executor.execute("denoise_image",
                             {"method": action["method"], "sigma": action["sigma"]})
            evaluation = executor.execute("evaluate_quality", {})
            output, trace = executor.current, executor.trace
        else:
            result = coordinator.process(noisy, image, action["sample_id"])
            output, trace = result["output"], result["trace"]
            evaluation = result["evaluation"]
        noisy_m = metrics.full_reference_metrics(image, noisy)
        used = [t["args"].get("method") for t in trace
                if t["tool"] == "denoise_image"]
        response["reply"] += (
            f"\n\n处理完成! 使用算法 **{' + '.join(m for m in used if m)}**: \n"
            f"- 噪声图 PSNR {noisy_m['psnr']}dB → 去噪后 **{evaluation.get('psnr', '-')}dB**\n"
            f"- SSIM {noisy_m['ssim']} → **{evaluation.get('ssim', '-')}**")
        response["images"] = [
            {"label": "原始图像", "src": _to_b64(image)},
            {"label": f"含噪输入 σ={action['sigma']:g}", "src": _to_b64(noisy)},
            {"label": "Agent处理结果", "src": _to_b64(output)},
        ]
    return response


@app.get("/api/paper_results")
def paper_results():
    path = ROOT / "experiments" / "paper_results.json"
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/benchmark")
def benchmark():
    if not BENCHMARK_CSV.exists():
        return {"rows": []}
    with open(BENCHMARK_CSV, encoding="utf-8") as f:
        return {"rows": list(csv.DictReader(f))}


@app.get("/")
def index():
    return FileResponse(ROOT / "frontend" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
