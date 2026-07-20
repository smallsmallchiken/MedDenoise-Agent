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
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from skimage import io as skio, transform
from skimage.util import img_as_float

from meddenoise import data as med_data
from meddenoise.agent.coordinator import Coordinator
from meddenoise.tools import metrics

ROOT = Path(__file__).resolve().parent
SET12_DIR = ROOT / "data" / "datasets" / "Set12"
BENCHMARK_CSV = ROOT / "experiments" / "results" / "benchmark.csv"

app = FastAPI(title="MedDenoise-Agent")
coordinator = Coordinator(verbose=False)


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
