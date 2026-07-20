"""决策规划器: 优先使用LLM(Anthropic Claude)推理, 无API Key时回退到专家规则引擎.

这保证系统在离线/答辩环境下也可完整演示, 同时保留LLM Agent的推理能力.
"""

import json
import os

from ..tools import dncnn


def rule_based_plan(perception: dict, similar_cases: list[dict] | None = None) -> list[dict]:
    """专家规则规划器: 根据感知结果生成"去噪+增强"工具调用计划."""
    sigma = perception.get("estimated_sigma", 15.0)
    edge_density = perception.get("edge_density", 0.1)
    modality = perception.get("modality", "MRI")

    if similar_cases:
        best = max(similar_cases,
                   key=lambda c: c.get("evaluation", {}).get("psnr", 0))
        if best.get("evaluation", {}).get("psnr", 0) > 30 and best.get("plan"):
            return best["plan"]

    plan: list[dict] = []
    if sigma < 3:
        pass  # 噪声极低, 无需去噪
    elif 18 <= sigma <= 32 and dncnn.is_available():
        # 中强噪声且接近DnCNN训练噪声水平(σ=25): 深度学习去噪器效果最佳
        plan.append({"tool": "denoise_image",
                     "args": {"method": "dncnn", "sigma": sigma}})
    elif edge_density > 0.12 or sigma > 25:
        # 结构复杂或强噪声: 结构感知VT-BM3D兼顾细节保留与去噪强度
        plan.append({"tool": "denoise_image",
                     "args": {"method": "vt-bm3d", "sigma": sigma}})
    elif sigma > 10:
        plan.append({"tool": "denoise_image",
                     "args": {"method": "bm3d", "sigma": sigma}})
    else:
        plan.append({"tool": "denoise_image",
                     "args": {"method": "nlm", "sigma": sigma}})

    if modality in ("X-Ray", "CT") or perception.get("dynamic_range", 1.0) < 0.7:
        plan.append({"tool": "enhance_image", "args": {"method": "clahe"}})
    if sigma > 10:
        plan.append({"tool": "enhance_image", "args": {"method": "unsharp"}})
    plan.append({"tool": "evaluate_quality", "args": {}})
    return plan


def llm_plan(perception: dict, skills_context: str = "") -> list[dict] | None:
    """调用Claude根据感知结果与领域技能生成处理计划; 失败返回None."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(
            api_key=api_key,
            base_url=os.environ.get("ANTHROPIC_BASE_URL") or None,
        )
        prompt = f"""你是医学影像去噪专家Agent。基于以下感知分析结果, 输出一个JSON数组形式的处理计划。
可用工具:
- denoise_image(method: gaussian/median/nlm/wavelet/bm3d/vt-bm3d/dncnn, sigma: number)
- enhance_image(method: clahe/unsharp/gamma)
- evaluate_quality()

领域知识:
{skills_context}

感知结果: {json.dumps(perception, ensure_ascii=False)}

只输出JSON数组, 每个元素形如 {{"tool": "...", "args": {{...}}}}, 最后一步必须是evaluate_quality。"""
        message = client.messages.create(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in message.content if b.type == "text")
        start, end = text.find("["), text.rfind("]") + 1
        plan = json.loads(text[start:end])
        if isinstance(plan, list) and all("tool" in step for step in plan):
            return plan
    except Exception:
        return None
    return None


def make_plan(perception: dict, similar_cases: list[dict] | None = None,
              skills_context: str = "") -> tuple[list[dict], str]:
    """返回(计划, 规划器类型). 优先LLM, 回退规则引擎."""
    plan = llm_plan(perception, skills_context)
    if plan is not None:
        return plan, "llm"
    return rule_based_plan(perception, similar_cases), "rule"
