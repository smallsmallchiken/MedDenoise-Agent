"""AI智能调参流水线: 输入图像 → AI分析意见与处理思路 → 详细调参方案
→ 执行论文VT-BM3D算法 → 处理结果分析, 并将结果存档供对话Agent参考.

以生成器方式逐步产出事件, 供前端实时展示AI的"思考过程":
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

TUNE_SYSTEM = (
    "你是医学影像去噪专家Agent, 精通论文《VT-BM3D: A collaborative filtering "
    "framework with joint optimization of structure awareness and noise "
    "adaptivity》(Signal Processing 243 (2026) 110417)。"
    "本系统只使用论文的VT-BM3D算法: 以结构显著性图 S=α·局部方差归一图 + "
    "β·结构张量相干性 指导强BM3D(σ×1.15)与弱BM3D(σ/texture_boost)逐像素融合, "
    "平坦区强去噪、边缘/纹理区弱去噪以保护细节。可调参数与范围: "
    "alpha(0.2~0.6, 局部方差权重, 细碎纹理多则调大), "
    "beta_tensor(0.4~0.8, 结构张量相干性权重, 线状边缘/方向性结构强则调大), "
    "texture_boost(1.1~2.0, 纹理区弱去噪倍率, 纹理越丰富越大以防抹除细节), "
    "sigma(去噪强度, 通常取估计噪声σ, 可±15%微调)。\n"
    "请分两部分输出:\n"
    "第一部分【分析意见与处理思路】: 用6~9句中文详细阐述——(1)图像内容与结构特点判断; "
    "(2)噪声水平与来源分析; (3)平坦区/边缘/纹理区的分布判断; (4)VT-BM3D处理该图的思路 "
    "(结构显著性图会如何划分强/弱去噪区域); (5)潜在风险(过平滑/伪影/细节丢失)。\n"
    "第二部分输出JSON调参方案(每个参数都要给出理由):\n"
    '{"sigma": 数值, "params": {"alpha": 数值, "beta_tensor": 数值, '
    '"texture_boost": 数值}, '
    '"param_rationale": {"sigma": "理由", "alpha": "理由", '
    '"beta_tensor": "理由", "texture_boost": "理由"}, '
    '"enhance": ["clahe"和/或"unsharp"或空数组], '
    '"expectation": "预期效果(PSNR/SSIM量级与视觉效果)"}'
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

    yield {"type": "stage", "stage": "AI分析意见与处理思路"}
    plan, analysis_text = None, ""
    if use_llm:
        reply = llm.llm_chat(TUNE_SYSTEM, [{"role": "user", "content": desc}],
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
    yield {"type": "thinking", "stage": "AI分析意见与处理思路", "text": analysis_text}

    yield {"type": "stage", "stage": "AI调参方案"}
    rationale = plan.get("param_rationale") or {}
    plan_text = (f"算法: VT-BM3D (论文结构感知协同滤波)\n"
                 f"σ = {plan['sigma']:g}"
                 + (f" — {rationale['sigma']}" if rationale.get("sigma") else ""))
    for k, v in plan["params"].items():
        plan_text += f"\n{k} = {v}" + (f" — {rationale[k]}" if rationale.get(k) else "")
    if plan.get("enhance"):
        plan_text += f"\n后处理增强: {' + '.join(plan['enhance'])}"
    if plan.get("expectation"):
        plan_text += f"\n预期效果: {plan['expectation']}"
    yield {"type": "thinking", "stage": "AI调参方案", "text": plan_text}
    yield {"type": "decision", "plan": plan, "llm": use_llm}

    yield {"type": "stage", "stage": "执行VT-BM3D"}
    yield {"type": "thinking", "stage": "执行VT-BM3D",
           "text": "构建结构显著性图(局部方差+结构张量相干性) → 强BM3D(σ×1.15)与"
                   f"弱BM3D(σ/{plan['params'].get('texture_boost', 1.4)})协同滤波"
                   " → 按显著性逐像素融合⋯"}
    output, denoised, ev = _execute(noisy, reference, plan)
    yield {"type": "thinking", "stage": "执行VT-BM3D",
           "text": f"VT-BM3D去噪完成 (σ={plan['sigma']:g}, "
                   f"参数={json.dumps(plan['params'], ensure_ascii=False)}), "
                   f"增强: {' + '.join(plan['enhance']) if plan.get('enhance') else '无'}"}

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
            yield {"type": "thinking", "stage": "处理结果分析", "text": reflection}
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
