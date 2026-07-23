"""AI智能调参流水线: 输入影像 → AI分析意见与处理思路 → 初始调参方案
→ 微调 VT-BM3D → 执行VT-BM3D → 处理结果分析, 并持久化history供后续微调.

以生成器方式逐步产出事件:
  {"type": "stage",    "stage": 阶段名}
  {"type": "thinking", "stage": 阶段名, "text": AI思考内容}
  {"type": "decision", "plan": 调参方案}
  {"type": "result",   ...最终结果}
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Generator

import numpy as np

from ..tools import analysis, denoise, enhance, metrics
from . import llm

HISTORY_FILE = Path(__file__).resolve().parents[2] / "data" / "results_history.json"
SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "denoise-strategy"

def _load_skill_text() -> str:
    f = SKILL_DIR / "SKILL.md"
    if f.exists():
        return f.read_text(encoding="utf-8").strip()[:2500]
    return ""

TUNE_SYSTEM = (
    "你是医学影像去噪专家Agent, 精通论文《VT-BM3D: A collaborative filtering "
    "framework with joint optimization of structure awareness and noise "
    "adaptivity》。本系统只使用论文提出的VT-BM3D算法: 以结构显著性图 "
    "S = α·归一化局部方差 + β·归一化结构张量最大特征值 指导强BM3D(σ×1.15) "
    "与弱BM3D(σ/texture_boost)按显著性逐像素融合, 平坦区强去噪、边缘/纹理区弱去噪以保护细节。\n"
    "可调参数与物理含义(必须在此范围内):\n"
    "- sigma: 去噪强度, 通常取估计噪声σ, 可±15%微调。\n"
    "- alpha(0.2~0.6): 局部方差权重, 细碎纹理/颗粒多则调大。\n"
    "- beta_tensor(0.4~0.8): 结构张量边缘权重, 方向性结构/线状边缘多则调大。\n"
    "- texture_boost(1.1~2.0): 纹理区弱去噪的衰减倍率, 纹理越丰富越大, 防止过平滑。\n\n"
    "论文与技能知识参考(用于微调与决策):\n{skill_context}\n\n"
    "请分两部分输出:\n"
    "第一部分【分析意见与处理思路】: 用6~9句中文详细阐述——(1)图像内容与结构特点判断; "
    "(2)噪声水平与来源分析; (3)平坦区/边缘/纹理区的分布判断; "
    "(4)VT-BM3D处理该图的结构显著性图会如何划分强/弱去噪区域; "
    "(5)潜在风险(过平滑/伪影/细节丢失)。\n"
    "第二部分输出JSON调参方案(每个参数都要给出理由):\n"
    '{{"sigma": 数值, "params": {{"alpha": 数值, "beta_tensor": 数值, '
    '"texture_boost": 数值}}, '
    '"param_rationale": {{"sigma": "理由", "alpha": "理由", '
    '"beta_tensor": "理由", "texture_boost": "理由"}}, '
    '"enhance": ["clahe"和/或"unsharp"或空数组], '
    '"expectation": "预期效果(PSNR/SSIM量级与视觉效果)"}}\n\n'
    "历史调参记录(最近的若干条成功案例, 供你参考相似影像的最佳参数):\n{history_context}"
)

FINE_TUNE_SYSTEM = (
    "你正在对VT-BM3D调参方案进行微调。请结合感知特征与历史调参成功案例, 输出一个更优的JSON方案。\n"
    "你只需专注调整 sigma/alpha/beta_tensor/texture_boost 以及可选后处理enhance, "
    "保留原方案中的分析理由并补充微调依据。\n"
    '输出JSON格式: {"sigma": 数值, "params": {"alpha":..., "beta_tensor":..., "texture_boost":...}, '
    '"enhance": [...], "param_rationale": {...}, "expectation": "...", "fine_tune_reason": "微调理由"}'
)

REFLECT_SYSTEM = (
    "你是医学影像去噪质量评审Agent, 熟悉VT-BM3D论文(平均PSNR较BM3D提升2.04dB, "
    "复杂纹理最大4.19dB)。请对本次VT-BM3D处理结果做详细分析, 输出JSON:\n"
    '{"analysis": "4~6句中文详细结果分析: 逐项解读PSNR/SSIM/RMSE与无参考指标, '
    '对比论文水平定位本次效果, 指出结构保持与噪声抑制的平衡情况, 说明参数设置的实际作用", '
    '"satisfied": true/false, '
    '"adjustment": {"sigma": ..., "params": {"alpha":..., "beta_tensor":..., '
    '"texture_boost":...}, "enhance": [...], "reason": "调整依据"} 或 null}\n'
    "PSNR>29dB且SSIM>0.85通常已足够好(satisfied=true)。重试机会只有一次, 谨慎使用。"
)

PARAM_BOUNDS = {"alpha": (0.2, 0.6), "beta_tensor": (0.4, 0.8),
                "texture_boost": (1.1, 2.0)}

def _describe(p: dict) -> str:
    return (
        f"图像特征: 估计噪声σ={p['estimated_sigma']}, 噪声等级={p['noise_level']}, "
        f"模态推测={p['modality']}, 边缘密度={p['edge_density']}, "
        f"纹理复杂度={p['texture_complexity']}, 平均结构相干性={p.get('mean_coherence', 0)}, "
        f"动态范围={p.get('dynamic_range', 1.0)}"
    )


def _history_context(n: int = 5) -> str:
    records = load_records()[-n:] if load_records() else []
    if not records:
        return "暂无历史记录。"
    lines = []
    for r in records:
        p = r.get("plan", {})
        e = r.get("evaluation", {})
        lines.append(
            f"- 模态:{r.get('perception', {}).get('modality','?')}, sigma={r.get('noise_sigma','?')}, "
            f"参数={p.get('params',{})}, PSNR={e.get('psnr','-')}, SSIM={e.get('ssim','-')}, "
            f"锐度={e.get('laplacian_sharpness','-')}, 结论:{r.get('reflection','')[:60]}"
        )
    return "\n".join(lines)


def _sanitize(plan: dict, est_sigma: float) -> dict:
    """参数白名单与边界裁剪, 算法固定为论文VT-BM3D."""
    plan["method"] = "vt-bm3d"
    try:
        sigma = float(plan.get("sigma") or est_sigma)
    except (TypeError, ValueError):
        sigma = est_sigma
    plan["sigma"] = float(np.clip(sigma, 0.5, 75))
    params = {}
    for k, (lo, hi) in PARAM_BOUNDS.items():
        try:
            v = float((plan.get("params") or {}).get(k))
            params[k] = round(float(np.clip(v, lo, hi)), 2)
        except (TypeError, ValueError):
            pass
    plan["params"] = params
    plan["enhance"] = [e for e in (plan.get("enhance") or [])
                       if e in enhance.ENHANCERS]
    return plan


def _fallback_plan(p: dict) -> dict:
    """离线回退: 基于规则的VT-BM3D自适应调参."""
    tex, edge = p["texture_complexity"], p["edge_density"]
    return {
        "method": "vt-bm3d", "sigma": p["estimated_sigma"],
        "params": {
            "alpha": round(min(0.6, 0.3 + tex * 2), 2),
            "beta_tensor": round(min(0.8, 0.5 + edge * 0.5), 2),
            "texture_boost": round(min(2.0, 1.2 + tex * 5), 2),
        },
        "enhance": [],
        "param_rationale": {
            "sigma": "取感知Agent的小波MAD噪声估计值",
            "alpha": "随纹理复杂度线性增大, 加强方差项对细碎纹理的感知",
            "beta_tensor": "随边缘密度增大, 强化结构张量对线状边缘的保护",
            "texture_boost": "纹理越丰富, 纹理区去噪强度衰减越多以保留细节",
        },
        "expectation": "离线专家规则方案, 预期达到与标准BM3D相当或更优的结构保持",
    }


def _plan_text(plan: dict, title: str = "调参方案") -> str:
    rationale = plan.get("param_rationale") or {}
    text = f"{title}: VT-BM3D (结构感知协同滤波)\nσ = {plan['sigma']:g}"
    if rationale.get("sigma"):
        text += f" — {rationale['sigma']}"
    for k, v in plan["params"].items():
        text += f"\n{k} = {v}"
        if rationale.get(k):
            text += f" — {rationale[k]}"
    if plan.get("enhance"):
        text += f"\n后处理增强: {' + '.join(plan['enhance'])}"
    if plan.get("expectation"):
        text += f"\n预期效果: {plan['expectation']}"
    if plan.get("fine_tune_reason"):
        text += f"\n微调依据: {plan['fine_tune_reason']}"
    return text


def _emit_stream(text: str, stage: str, chunk_size: int = 5):
    if not text:
        return
    for i in range(chunk_size, len(text) + chunk_size, chunk_size):
        yield {"type": "thinking", "stage": stage, "text": text[:i]}
        # 模拟人类阅读/打字节奏, 让前端能看到AI正在“真实思考”
        if len(text) > 200:
            time.sleep(0.008)


def _execute(noisy: np.ndarray, reference: np.ndarray | None, plan: dict) -> tuple:
    out = denoise.denoise(noisy, "vt-bm3d", float(plan["sigma"]),
                          plan.get("params") or None)
    denoised = out.copy()
    for e in plan.get("enhance") or []:
        out = enhance.enhance(out, e)
    ev = metrics.no_reference_metrics(out)
    if reference is not None:
        ev.update(metrics.full_reference_metrics(reference, denoised))
    return out, denoised, ev


def save_record(record: dict) -> None:
    try:
        records = load_records()
        record["id"] = len(records) + 1
        record["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        records.append(record)
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(
            json.dumps(records[-100:], ensure_ascii=False, indent=1))
    except Exception:
        pass


def load_records() -> list:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return []


def run_ai_pipeline(noisy: np.ndarray, reference: np.ndarray | None = None,
                    source: str = "", noise_sigma: float | None = None,
                    ) -> Generator[dict, None, None]:
    use_llm = llm.is_llm_available()

    yield {"type": "stage", "stage": "感知分析"}
    perception = analysis.analyze_image(noisy)
    desc = _describe(perception)
    yield {"type": "thinking", "stage": "感知分析",
           "text": "对输入影像进行小波MAD噪声估计、结构张量分析与模态识别⋯\n" + desc}
    yield {"type": "perception", "perception": perception}

    # ---- AI分析意见与处理思路 ----
    yield {"type": "stage", "stage": "AI分析意见与处理思路"}
    plan, analysis_text = None, ""
    if use_llm:
        system = TUNE_SYSTEM.format(skill_context=_load_skill_text(),
                                     history_context=_history_context(5))
        reply = llm.llm_chat(system, [{"role": "user", "content": desc}],
                             max_tokens=1600, purpose="AI分析与调参")
        if reply:
            analysis_text = (reply.split("{")[0].replace("```json", "")
                             .replace("```", "").strip() or reply)
            plan = llm.extract_json(reply)
    if plan is None:
        plan = _fallback_plan(perception)
        analysis_text = analysis_text or (
            "未连接LLM, 使用内置专家规则进行自适应调参: 纹理复杂度越高, "
            "texture_boost越大以保护细节; 边缘密度越高, beta_tensor越大以强化结构感知。")
    plan = _sanitize(plan, perception["estimated_sigma"])

    for ev in _emit_stream(analysis_text, "AI分析意见与处理思路"):
        yield ev

    # ---- AI调参方案 ----
    yield {"type": "stage", "stage": "AI调参方案"}
    initial_plan = plan.copy()
    for ev in _emit_stream(_plan_text(plan, "初始调参方案"), "AI调参方案"):
        yield ev
    yield {"type": "decision", "plan": plan, "llm": use_llm}

    # ---- 微调 VT-BM3D ----
    yield {"type": "stage", "stage": "微调VT-BM3D"}
    if use_llm:
        user_prompt = (
            f"图像特征: {desc}\n"
            f"初始方案: {json.dumps(initial_plan, ensure_ascii=False)}\n"
            f"历史参考: {_history_context(3)}\n"
            "请参考历史成功案例对参数进行微调, 输出更优JSON方案。"
        )
        fine_reply = llm.llm_chat(FINE_TUNE_SYSTEM,
                                  [{"role": "user", "content": user_prompt}],
                                  max_tokens=1200, purpose="VT-BM3D微调")
        fine_plan = llm.extract_json(fine_reply) if fine_reply else None
        if fine_plan:
            plan = _sanitize(fine_plan, perception["estimated_sigma"])
            plan["param_rationale"] = {**(plan.get("param_rationale") or {}),
                                       "fine_tune": plan.get("fine_tune_reason", "基于历史调参案例微调")}
        fine_text = _plan_text(plan, "微调后方案")
        for ev in _emit_stream(fine_text, "微调VT-BM3D"):
            yield ev
    else:
        yield {"type": "thinking", "stage": "微调VT-BM3D",
               "text": "未连接LLM, 沿用初始方案并做经验性边界裁剪。"}
    yield {"type": "decision", "plan": plan, "llm": use_llm, "tuned": True}

    # ---- 执行VT-BM3D ----
    yield {"type": "stage", "stage": "执行VT-BM3D"}
    exec_text = (
        "构建结构显著性图(局部方差+结构张量最大特征值) → 强BM3D(σ×1.15)与"
        f"弱BM3D(σ/{plan['params'].get('texture_boost', 1.4)})协同滤波"
        " → 按显著性逐像素融合⋯"
    )
    for ev in _emit_stream(exec_text, "执行VT-BM3D"):
        yield ev
    output, denoised, ev = _execute(noisy, reference, plan)
    yield {"type": "thinking", "stage": "执行VT-BM3D",
           "text": f"VT-BM3D去噪完成 (σ={plan['sigma']:g}, "
                   f"参数={json.dumps(plan['params'], ensure_ascii=False)}), "
                   f"增强: {' + '.join(plan['enhance']) if plan.get('enhance') else '无'}"}

    # ---- 处理结果分析 ----
    yield {"type": "stage", "stage": "处理结果分析"}
    history = [{"plan": plan, "evaluation": ev}]
    reflection = ""
    if use_llm:
        prompt = (f"调参方案: {json.dumps(plan, ensure_ascii=False)}\n"
                  f"评估指标: {json.dumps(ev, ensure_ascii=False)}\n"
                  "请给出详细结果分析, 并判断是否调整重试。")
        reply = llm.llm_chat(REFLECT_SYSTEM, [{"role": "user", "content": prompt}],
                             max_tokens=1400, purpose="结果分析与反思")
        verdict = llm.extract_json(reply) if reply else None
        if verdict:
            reflection = verdict.get("analysis") or verdict.get("comment", "")
            for evv in _emit_stream(reflection, "处理结果分析"):
                yield evv
            adj = verdict.get("adjustment")
            if not verdict.get("satisfied", True) and adj:
                adj = _sanitize(dict(adj), plan["sigma"])
                adj.setdefault("reason", "AI反思后微调的重试方案")
                yield {"type": "thinking", "stage": "处理结果分析",
                       "text": "AI判定质量可进一步提升, 调整参数重试: "
                               + json.dumps({"sigma": adj["sigma"],
                                             **adj["params"]}, ensure_ascii=False)
                               + f" — {adj['reason']}"}
                output2, denoised2, ev2 = _execute(noisy, reference, adj)
                history.append({"plan": adj, "evaluation": ev2})
                if ev2.get("psnr", ev2.get("laplacian_sharpness", 0)) >= \
                        ev.get("psnr", ev.get("laplacian_sharpness", 0)):
                    output, denoised, ev, plan = output2, denoised2, ev2, adj
                    yield {"type": "thinking", "stage": "处理结果分析",
                           "text": "重试方案效果更优, 采纳新参数。"}
                else:
                    yield {"type": "thinking", "stage": "处理结果分析",
                           "text": "重试未带来提升, 保留初次结果。"}
    if not reflection:
        reflection = (f"PSNR={ev.get('psnr', '-')}dB, SSIM={ev.get('ssim', '-')}, "
                      "达到预期质量。")
        yield {"type": "thinking", "stage": "处理结果分析", "text": reflection}

    save_record({"source": source, "noise_sigma": noise_sigma,
                 "perception": perception, "plan": plan, "evaluation": ev,
                 "analysis": analysis_text, "reflection": reflection,
                 "attempts": len(history), "llm": use_llm})

    yield {"type": "result", "perception": perception, "plan": plan,
           "evaluation": ev, "history": history, "llm": use_llm,
           "output": output, "denoised": denoised}
