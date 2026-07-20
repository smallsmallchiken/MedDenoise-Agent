"""对话Agent: 以自然语言方式与医学影像去噪系统交互.

参考 emperor-agent 的"对话即工作台"思路: 用户通过中文对话即可
调度感知/决策/执行/评估全流程, 查询论文结论与实验数据,
了解算法原理。无需API Key, 内置中文意图解析器; 配置
ANTHROPIC_API_KEY 后自动升级为LLM对话。
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ALGO_DOCS = {
    "vt-bm3d": (
        "**VT-BM3D (方差-张量BM3D)** 是本系统的核心算法, 来自论文 "
        "《VT-BM3D: A collaborative filtering framework with joint optimization "
        "of structure awareness and noise characteristics》(Signal Processing 243, 2026)。\n\n"
        "核心思想: \n"
        "1. 计算局部方差图 V 与结构张量相干性图 C\n"
        "2. 构建结构显著性图 S = 0.4·V + 0.6·C\n"
        "3. 分别以强(1.15σ)/弱(σ/1.4)两档运行BM3D\n"
        "4. 逐像素融合: 平坦区强去噪、边缘纹理区弱去噪\n\n"
        "论文结论: 相对BM3D平均PSNR提升 **2.04dB**, 复杂纹理最大提升 **4.19dB**。"
    ),
    "dncnn": (
        "**DnCNN** 是本系统的深度学习去噪器 (PyTorch实现)。\n\n"
        "它由17层卷积组成, 采用**残差学习**: 网络学习预测噪声残差 v, "
        "再用 y - v 得到干净图像; 配合批归一化加速收敛。"
        "本系统加载σ=25高斯噪声预训练权重, 在σ=18~32噪声区间由决策Agent优先选用, "
        "实测Set12上可将PSNR从约20dB提升到30dB。"
    ),
    "bm3d": (
        "**BM3D** 是经典的块匹配三维协同滤波算法 (Dabov et al., IEEE TIP 2007): "
        "将相似图像块堆叠为3D数组, 在变换域做协同硬阈值/维纳滤波, "
        "长期作为传统去噪方法的基准。其缺点是全图使用固定参数, "
        "细节丰富区域容易过度平滑 —— 这正是VT-BM3D改进的出发点。"
    ),
    "nlm": "**非局部均值 (NLM)**: 利用图像自相似性, 以相似邻域加权平均去噪, 适合轻度噪声。",
    "wavelet": "**小波阈值去噪**: 在小波域对系数做软阈值收缩, 速度快, 适合中低噪声。",
    "median": "**中值滤波**: 取邻域中值, 对椒盐噪声特别有效。",
    "gaussian": "**高斯滤波**: 简单的低通平滑, 速度最快但会模糊边缘, 常作对比基线。",
}

HELP_TEXT = (
    "你好, 我是医学影像去噪Agent, 可以直接用中文和我对话: \n\n"
    "- **处理图像**: 如「用σ=25处理Set12第5张」「用vt-bm3d处理CT体模」\n"
    "- **算法原理**: 如「介绍一下VT-BM3D」「DnCNN是什么」\n"
    "- **论文结论**: 如「论文里VT-BM3D比BM3D好多少」\n"
    "- **实验数据**: 如「σ=25下哪个算法最好」\n"
    "- **系统架构**: 如「你是怎么决策的」\n\n"
    "也可以先在左侧上传图像, 再让我处理。"
)

ARCH_TEXT = (
    "我采用**多Agent协作架构**, 处理一张影像的完整流程是: \n\n"
    "1. **感知Agent**: 小波MAD噪声估计、结构张量分析、模态识别(CT/MRI/X光/超声)\n"
    "2. **决策Agent**: 结合技能知识库与案例记忆制定方案 —— "
    "σ在18~32且DnCNN权重可用时优先深度学习, 强噪声或复杂结构选VT-BM3D, "
    "中等噪声用BM3D, 轻噪声用NLM\n"
    "3. **执行Agent**: 依次调用去噪、CLAHE/锐化增强等工具\n"
    "4. **评估Agent**: 计算PSNR/SSIM与无参考指标, 不达标时触发**反思重规划**\n\n"
    "配置 ANTHROPIC_API_KEY 后决策由LLM完成, 否则使用内置专家规则, 离线也能完整运行。"
)


def _load_benchmark() -> list[dict]:
    path = ROOT / "experiments" / "results" / "benchmark.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_paper() -> dict:
    path = ROOT / "experiments" / "paper_results.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_sigma(text: str) -> float | None:
    m = re.search(r"(?:σ|sigma|噪声|西格玛)\s*[=为是]?\s*(\d+(?:\.\d+)?)", text, re.I)
    return float(m.group(1)) if m else None


def _parse_method(text: str) -> str | None:
    lower = text.lower().replace("_", "-")
    for name in ("vt-bm3d", "vtbm3d", "dncnn", "bm3d", "nlm", "wavelet", "median", "gaussian"):
        if name in lower:
            return "vt-bm3d" if name == "vtbm3d" else name
    for zh, name in (("小波", "wavelet"), ("中值", "median"), ("高斯滤波", "gaussian"),
                     ("非局部", "nlm"), ("深度学习", "dncnn"), ("神经网络", "dncnn")):
        if zh in text:
            return name
    return None


def _parse_sample(text: str) -> str | None:
    if "体模" in text or "ct" in text.lower() and "体" in text:
        return "ct_phantom"
    m = re.search(r"set\s*12\s*[#第·]?\s*(\d{1,2})", text, re.I)
    if m:
        return f"set12_{int(m.group(1)):02d}"
    return None


class ChatAgent:
    """中文意图解析对话Agent."""

    def reply(self, text: str) -> dict:
        """解析用户消息, 返回 {reply, action?}.

        action = {"type": "process", "sample_id", "sigma", "method"}
        由服务端在收到action后执行完整Agent流程。
        """
        t = text.strip()
        low = t.lower()

        if not t or re.fullmatch(r"(你好|您好|hi|hello|在吗|帮助|help|你能做什么|你会什么)[!！?？。~]*", low):
            return {"reply": HELP_TEXT}

        if any(k in t for k in ("架构", "怎么决策", "如何决策", "工作原理", "决策流程", "怎么工作")):
            return {"reply": ARCH_TEXT}

        if any(k in t for k in ("论文", "文献", "paper")):
            paper = _load_paper()
            s = paper["summary"]
            rows = [r for r in paper["table2"] if r["sigma"] == 15]
            lines = [f"- {r['dataset']} {r['noise']}噪声 σ=15: BM3D {r['bm3d_psnr']}dB → "
                     f"VT-BM3D {r['vtbm3d_psnr']}dB (**+{r['delta_psnr']}dB**)" for r in rows]
            return {"reply": (
                f"论文《VT-BM3D》(Signal Processing 243, 2026) 的核心结论: \n\n"
                f"- 平均PSNR提升 **{s['avg_psnr_gain_db']}dB**, 复杂纹理最大 **{s['max_psnr_gain_db']}dB**\n"
                + "\n".join(lines)
                + "\n\n完整数据见「论文结果」页, 图表可直接用于论文与答辩。")}

        method = _parse_method(t)
        if any(k in t for k in ("介绍", "是什么", "原理", "解释", "说说", "讲讲")) and method:
            return {"reply": ALGO_DOCS[method]}

        if any(k in t for k in ("实验", "评测", "哪个算法", "对比", "最好")):
            rows = _load_benchmark()
            if not rows:
                return {"reply": "本地实验数据还没有生成, 请先运行 `python experiments/run_experiments.py`。"}
            sigma = _parse_sigma(t) or 25
            cand = [r for r in rows if abs(float(r["sigma"]) - sigma) < 1e-6 and r["method"] != "noisy"]
            if not cand:
                return {"reply": f"本地实验没有 σ={sigma:g} 的数据, 已有 σ=15/25/35。"}
            cand.sort(key=lambda r: -float(r["psnr"]))
            lines = [f"| {r['method']} | {r['psnr']} | {r['ssim']} |" for r in cand]
            best = cand[0]
            return {"reply": (
                f"Set12+BSD300 数据集 σ={sigma:g} 高斯噪声下的实测结果 (平均值): \n\n"
                "| 算法 | PSNR(dB) | SSIM |\n| --- | --- | --- |\n"
                + "\n".join(lines)
                + f"\n\n表现最好的是 **{best['method']}** ({best['psnr']}dB)。")}

        if any(k in t for k in ("处理", "去噪", "试试", "跑一下", "运行", "演示")):
            sample = _parse_sample(t) or "set12_05"
            sigma = _parse_sigma(t) or 25.0
            return {
                "reply": (
                    f"好的, 我来处理 **{'CT体模' if sample == 'ct_phantom' else 'Set12 #' + sample[-2:]}**"
                    f" (合成噪声 σ={sigma:g}"
                    + (f", 指定算法 {method}" if method else ", 算法由决策Agent自动选择")
                    + ") ⋯"),
                "action": {"type": "process", "sample_id": sample,
                           "sigma": sigma, "method": method},
            }

        if method:
            return {"reply": ALGO_DOCS[method]}

        return {"reply": (
            "我没有完全理解这句话。" + HELP_TEXT)}
