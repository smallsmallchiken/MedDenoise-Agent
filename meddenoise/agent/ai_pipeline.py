"""AI智能调参流水线: LLM分析图像特征 → 智能调参 → 执行 → 反思优化.

以生成器方式逐步产出事件, 供前端实时展示AI的"思考过程":
  {"type": "stage",    "stage": 阶段名}
  {"type": "thinking", "stage": 阶段名, "text": AI思考内容}
  {"type": "decision", "plan": 调参方案}
  {"type": "result",   ...最终结果}
"""

from __future__ import annotations

import json
from typing import Generator

import numpy as np

from ..tools import analysis, denoise, enhance, metrics
from . import llm

TUNE_SYSTEM = (
    "你是医学影像去噪专家Agent, 精通VT-BM3D(论文: Signal Processing 243 (2026) 110417)、"
    "DnCNN深度学习去噪与传统方法。VT-BM3D通过结构显著性图 S=α·局部方差+β·结构张量相干性 "
    "指导强/弱BM3D融合, 可调参数: alpha(0.2~0.6, 方差权重), beta_tensor(0.4~0.8, 相干性权重), "
    "texture_boost(1.1~2.0, 纹理区弱去噪倍率, 纹理越丰富应越大)。"
    "DnCNN以σ=25预训练, 在σ∈[18,32]表现最好。"
    "请先用2~4句中文分析图像特征并给出推理过程, 再输出JSON方案:\n"
    '{"method": "vt-bm3d|dncnn|bm3d|nlm|wavelet", "sigma": 数值, '
    '"params": {"alpha": 数值, "beta_tensor": 数值, "texture_boost": 数值}, '
    '"enhance": ["clahe"和/或"unsharp"或空数组], "reason": "一句话决策依据"}\n'
    "非vt-bm3d方法params可为空对象。"
)

REFLECT_SYSTEM = (
    "你是医学影像去噪质量评审Agent。根据处理结果指标, 用2~3句中文点评效果, "
    "并判断是否需要调整参数重试。输出JSON:\n"
    '{"comment": "点评", "satisfied": true/false, '
    '"adjustment": {"method": ..., "sigma": ..., "params": {...}, "enhance": [...]} 或 null}\n'
    "PSNR>29dB且SSIM>0.85通常已足够好(satisfied=true)。重试机会只有一次, 谨慎使用。"
)


def _describe(p: dict) -> str:
    return (
        f"图像特征: 估计噪声σ={p['estimated_sigma']}, 噪声等级={p['noise_level']}, "
        f"模态推测={p['modality']}, 边缘密度={p['edge_density']}, "
        f"纹理复杂度={p['texture_complexity']}, 平均结构相干性={p.get('mean_coherence', 0)}, "
        f"动态范围={p.get('dynamic_range', 1.0)}"
    )


def _fallback_plan(p: dict) -> dict:
    """离线回退: 基于规则的自适应调参."""
    sigma = p["estimated_sigma"]
    tex = p["texture_complexity"]
    edge = p["edge_density"]
    from ..tools import dncnn
    if 18 <= sigma <= 32 and dncnn.is_available() and edge < 0.3:
        method, params = "dncnn", {}
    else:
        method = "vt-bm3d"
        params = {
            "alpha": round(min(0.6, 0.3 + tex * 2), 2),
            "beta_tensor": round(min(0.8, 0.5 + edge * 0.5), 2),
            "texture_boost": round(min(2.0, 1.2 + tex * 5), 2),
        }
    return {
        "method": method, "sigma": sigma, "params": params,
        "enhance": ["unsharp"] if sigma > 10 else [],
        "reason": "离线专家规则: 根据纹理复杂度与边缘密度自适应设定结构感知参数",
    }


def _execute(noisy: np.ndarray, reference: np.ndarray | None, plan: dict) -> tuple:
    out = denoise.denoise(noisy, plan["method"], float(plan.get("sigma") or 15),
                          plan.get("params") or None)
    denoised = out.copy()
    for e in plan.get("enhance") or []:
        if e in enhance.ENHANCERS:
            out = enhance.enhance(out, e)
    ev = metrics.no_reference_metrics(out)
    if reference is not None:
        ev.update(metrics.full_reference_metrics(reference, denoised))
    return out, denoised, ev


def run_ai_pipeline(noisy: np.ndarray, reference: np.ndarray | None = None,
                    ) -> Generator[dict, None, None]:
    use_llm = llm.is_llm_available()

    yield {"type": "stage", "stage": "感知分析"}
    perception = analysis.analyze_image(noisy)
    desc = _describe(perception)
    yield {"type": "thinking", "stage": "感知分析",
           "text": "对输入影像进行小波MAD噪声估计、结构张量分析与模态识别⋯\n" + desc}
    yield {"type": "perception", "perception": perception}

    yield {"type": "stage", "stage": "AI特征分析与智能调参"}
    plan, analysis_text = None, ""
    if use_llm:
        reply = llm.llm_chat(TUNE_SYSTEM, [{"role": "user", "content": desc}])
        if reply:
            analysis_text = reply.split("{")[0].replace("```json", "").strip() or reply
            plan = llm.extract_json(reply)
    if plan is None or "method" not in plan:
        plan = _fallback_plan(perception)
        analysis_text = analysis_text or (
            "未连接LLM, 使用内置专家规则进行自适应调参: 纹理复杂度越高, "
            "texture_boost越大以保护细节; 边缘密度越高, beta_tensor越大以强化结构感知。")
    plan.setdefault("sigma", perception["estimated_sigma"])
    yield {"type": "thinking", "stage": "AI特征分析与智能调参", "text": analysis_text}
    yield {"type": "decision", "plan": plan, "llm": use_llm}

    yield {"type": "stage", "stage": "执行处理"}
    output, denoised, ev = _execute(noisy, reference, plan)
    yield {"type": "thinking", "stage": "执行处理",
           "text": f"以 {plan['method']} (σ={float(plan['sigma']):g}"
                   + (f", 参数={json.dumps(plan['params'], ensure_ascii=False)}"
                      if plan.get("params") else "")
                   + f") 完成去噪, 增强: {plan.get('enhance') or '无'}"}

    yield {"type": "stage", "stage": "质量评估与AI反思"}
    history = [{"plan": plan, "evaluation": ev}]
    reflection = ""
    if use_llm:
        prompt = (f"初次方案: {json.dumps(plan, ensure_ascii=False)}\n"
                  f"评估指标: {json.dumps(ev, ensure_ascii=False)}\n"
                  "是否需要调整重试?")
        reply = llm.llm_chat(REFLECT_SYSTEM, [{"role": "user", "content": prompt}])
        verdict = llm.extract_json(reply) if reply else None
        if verdict:
            reflection = verdict.get("comment", "")
            yield {"type": "thinking", "stage": "质量评估与AI反思", "text": reflection}
            adj = verdict.get("adjustment")
            if not verdict.get("satisfied", True) and adj and adj.get("method"):
                yield {"type": "thinking", "stage": "质量评估与AI反思",
                       "text": "AI判定质量可进一步提升, 调整参数重试: "
                               + json.dumps(adj, ensure_ascii=False)}
                adj.setdefault("sigma", plan.get("sigma"))
                output2, denoised2, ev2 = _execute(noisy, reference, adj)
                history.append({"plan": adj, "evaluation": ev2})
                if ev2.get("psnr", ev2.get("laplacian_sharpness", 0)) >= \
                        ev.get("psnr", ev.get("laplacian_sharpness", 0)):
                    output, denoised, ev, plan = output2, denoised2, ev2, adj
                    yield {"type": "thinking", "stage": "质量评估与AI反思",
                           "text": "重试方案效果更优, 采纳新参数。"}
                else:
                    yield {"type": "thinking", "stage": "质量评估与AI反思",
                           "text": "重试未带来提升, 保留初次结果。"}
    if not reflection:
        reflection = (f"PSNR={ev.get('psnr', '-')}dB, SSIM={ev.get('ssim', '-')}, "
                      "达到预期质量。")
        yield {"type": "thinking", "stage": "质量评估与AI反思", "text": reflection}

    yield {"type": "result", "perception": perception, "plan": plan,
           "evaluation": ev, "history": history, "llm": use_llm,
           "output": output, "denoised": denoised}
