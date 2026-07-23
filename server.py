"""MedDenoise-Agent 前端后端服务 (FastAPI).

运行: python server.py  然后浏览器访问 http://localhost:8000
"""

import base64
import csv
import hashlib
import io
import json
import secrets
import time
from pathlib import Path

import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()
from skimage import io as skio, transform
from skimage.util import img_as_float

from meddenoise import data as med_data
from meddenoise.agent import llm
from meddenoise.agent.ai_pipeline import load_records, run_ai_pipeline
from meddenoise.agent.chat import ChatAgent
from meddenoise.agent.coordinator import Coordinator
from meddenoise.tools import metrics
from meddenoise.tools.registry import ToolExecutor

ROOT = Path(__file__).resolve().parent
MEDS_DIR = ROOT / "data" / "datasets" / "MedSmall"
BENCHMARK_CSV = ROOT / "experiments" / "results" / "benchmark.csv"
USERS_FILE = ROOT / "data" / "users.json"
TOKENS_FILE = ROOT / "data" / "tokens.json"

app = FastAPI(title="MedDenoise-Agent")
coordinator = Coordinator(verbose=False)
chat_agent = ChatAgent()

# ---- auth state ----
users: dict[str, str] = {}
tokens: dict[str, str] = {}


def _hash(pwd: str) -> str:
    return hashlib.sha256(pwd.encode("utf-8")).hexdigest()


def _load_auth() -> None:
    global users, tokens
    try:
        users = json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        users = {}
    try:
        tokens = json.loads(TOKENS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        tokens = {}
    if "admin" not in users:
        users["admin"] = _hash("123456")
        _save_users()


def _save_users() -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, ensure_ascii=False, indent=1),
                          encoding="utf-8")


def _save_tokens() -> None:
    TOKENS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKENS_FILE.write_text(json.dumps(tokens, ensure_ascii=False, indent=1),
                           encoding="utf-8")


_load_auth()


def _require_auth(authorization: str = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未登录")
    token = authorization[7:].strip()
    if token not in tokens:
        raise HTTPException(status_code=401, detail="登录已过期")
    return tokens[token]


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


MED_NAME_MAP = {
    "ct_phantom": "CT体模",
    "cell": "细胞显微",
    "mitosis": "有丝分裂",
    "microaneurysms": "微动脉瘤",
    "skin": "皮肤切片",
    "retina": "眼底视网膜",
    "immunohistochemistry": "免疫组化",
    "brain_slice": "脑部切片",
    "kidney_slice": "肾脏切片",
}


@app.get("/api/samples")
def list_samples(_: str = Depends(_require_auth)):
    samples = [{"id": f"med_{f.stem}",
                "name": MED_NAME_MAP.get(f.stem, f.stem)}
               for f in sorted(MEDS_DIR.glob("*.png"))]
    return samples


def _load_sample(sample_id: str) -> np.ndarray:
    if sample_id.startswith("med_"):
        path = MEDS_DIR / f"{sample_id[4:]}.png"
        if path.exists():
            return med_data.load_image(str(path))
    if sample_id == "ct_phantom":
        return med_data.load_sample_images()["ct_phantom"]
    raise ValueError(f"未知示例: {sample_id}")


@app.post("/api/process")
async def process(file: UploadFile | None = File(None),
                  sample_id: str = Form(""),
                  add_noise: bool = Form(True),
                  sigma: float = Form(25.0),
                  _: str = Depends(_require_auth)):
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
                     sigma: float = Form(25.0),
                     __: str = Depends(_require_auth)):
    if file is not None and file.filename:
        raw = skio.imread(io.BytesIO(await file.read()))
        image = _prepare(raw)
        source = file.filename
    elif sample_id:
        image = _load_sample(sample_id)
        source = sample_id
    else:
        return {"error": "请上传图像或选择示例"}

    reference = None
    noisy = image
    if add_noise:
        reference = image
        noisy = med_data.add_gaussian_noise(image, sigma=sigma)

    def streamer():
        for event in run_ai_pipeline(noisy, reference, source=source,
                                     noise_sigma=sigma if add_noise else None):
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
                    "tuned": event.get("llm", False),
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

    return StreamingResponse(streamer(), media_type="text/event-stream")


@app.get("/api/history")
def get_history(_: str = Depends(_require_auth)):
    return list(reversed(load_records()))[:30]


@app.get("/api/api_stats")
def api_stats(_: str = Depends(_require_auth)):
    return llm.usage_stats()


@app.post("/api/check_model")
def check_model(_: str = Depends(_require_auth)):
    return llm.check_reachability()


# ---- auth endpoints (public) ----
@app.post("/api/auth/register")
async def auth_register(username: str = Form(...),
                        password: str = Form(...),
                        confirm: str = Form(...),
                        email: str = Form(''),
                        code: str = Form('')):
    if password != confirm:
        return {"ok": False, "detail": "两次输入密码不一致"}
    if not username or not password:
        return {"ok": False, "detail": "用户名或密码不能为空"}
    _load_auth()
    if username in users:
        return {"ok": False, "detail": "用户名已存在"}
    users[username] = _hash(password)
    _save_users()
    return {"ok": True, "detail": "注册成功, 请登录"}


@app.post("/api/auth/login")
async def auth_login(username: str = Form(...), password: str = Form(...)):
    _load_auth()
    if username not in users or users[username] != _hash(password):
        return {"ok": False, "detail": "用户名或密码错误"}
    token = secrets.token_urlsafe(24)
    tokens[token] = username
    _save_tokens()
    return {"ok": True, "token": token, "username": username}


@app.post("/api/auth/logout")
async def auth_logout(authorization: str = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
        tokens.pop(token, None)
        _save_tokens()
    return {"ok": True}


@app.post("/api/auth/reset")
async def auth_reset(username: str = Form(...),
                     password: str = Form(...),
                     confirm: str = Form(...),
                     email: str = Form(''),
                     code: str = Form('')):
    if password != confirm:
        return {"ok": False, "detail": "两次输入密码不一致"}
    _load_auth()
    if username not in users:
        return {"ok": False, "detail": "用户名不存在"}
    users[username] = _hash(password)
    _save_users()
    # invalidate existing tokens for this user
    for t, u in list(tokens.items()):
        if u == username:
            tokens.pop(t, None)
    _save_tokens()
    return {"ok": True, "detail": "密码已重置, 请重新登录"}


@app.get("/api/auth/me")
async def auth_me(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        return {"ok": False}
    token = authorization[7:].strip()
    if token not in tokens:
        return {"ok": False}
    return {"ok": True, "username": tokens[token]}


# ---- model / api config (protected) ----
@app.get("/api/models")
def list_models(_: str = Depends(_require_auth)):
    return llm.list_models()


@app.get("/api/api_config")
def get_api_config(_: str = Depends(_require_auth)):
    cfg = llm.get_api_config()
    return {
        "model": cfg.get("model"),
        "base_url": cfg.get("base_url") or "官方默认",
        "max_tokens": cfg.get("max_tokens"),
        "configured": llm.is_llm_available(),
    }


@app.post("/api/api_config")
async def set_api_config(model: str = Form(...),
                         max_tokens: int = Form(1200),
                         _: str = Depends(_require_auth)):
    try:
        max_tokens = int(max_tokens)
        if max_tokens < 200:
            max_tokens = 200
        if max_tokens > 8000:
            max_tokens = 8000
    except (TypeError, ValueError):
        max_tokens = 1200
    llm.save_api_config({"model": model, "max_tokens": max_tokens})
    return {"ok": True}


CHAT_SYSTEM = (
    "你是「MedDenoise-Agent」医学影像智能去噪系统的AI助手, 精通VT-BM3D论文"
    "(Peng et al., Signal Processing 243 (2026) 110417: 结构感知+噪声自适应联合优化, "
    "相对BM3D平均PSNR提升2.04dB, 复杂纹理最大4.19dB)。"
    "本系统主流程只使用论文VT-BM3D算法: 结构显著性图 S=α·局部方差+β·结构张量相干性 "
    "指导强/弱BM3D逐像素融合; AI智能体负责分析图像特征并对σ/alpha/beta_tensor/"
    "texture_boost详细调参, 执行后给出结果分析并存档。数据集已替换为小型医学图像。"
    "回答可引用下方的最近处理记录进行分析对比。"
    "用简洁专业的中文回答, 可用Markdown, 专业名词(PSNR/SSIM等)保留英文。"
    "若用户想处理图像, 建议其直接说「用σ=25处理眼底视网膜」这样的指令。"
)


def _history_context() -> str:
    records = load_records()[-5:]
    if not records:
        return ""
    lines = ["\n【本系统最近的VT-BM3D处理记录(可供引用分析)】"]
    for r in records:
        ev, plan = r.get("evaluation", {}), r.get("plan", {})
        lines.append(
            f"#{r.get('id')} {r.get('time')} 图像={r.get('source')} "
            f"加噪σ={r.get('noise_sigma')} 调参={json.dumps(plan.get('params', {}))} "
            f"去噪σ={plan.get('sigma')} PSNR={ev.get('psnr', '-')} "
            f"SSIM={ev.get('ssim', '-')} 尝试次数={r.get('attempts')}")
    return "\n".join(lines)


@app.post("/api/chat")
async def chat(message: str = Form(...), history: str = Form("[]"),
               _: str = Depends(_require_auth)):
    out = chat_agent.reply(message)
    action = out.get("action")
    if action is None and llm.is_llm_available():
        try:
            msgs = [m for m in json.loads(history) if m.get("content")][-8:]
        except json.JSONDecodeError:
            msgs = []
        msgs.append({"role": "user", "content": message})
        reply = llm.llm_chat(CHAT_SYSTEM + _history_context(), msgs)
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
def paper_results(_: str = Depends(_require_auth)):
    path = ROOT / "experiments" / "paper_results.json"
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/benchmark")
def benchmark(_: str = Depends(_require_auth)):
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
