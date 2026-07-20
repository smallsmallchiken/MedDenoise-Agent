"""处理报告生成: 输出Markdown格式的智能诊断报告."""

import json
from datetime import datetime


def generate_report(result: dict, filename: str = "") -> str:
    perception = result["perception"]
    evaluation = result["evaluation"]
    lines = [
        "# 医学影像智能去噪处理报告",
        "",
        f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- 输入文件: {filename or '(内存图像)'}",
        f"- 规划器: {'LLM智能规划' if result['planner'] == 'llm' else '专家规则规划'}",
        "",
        "## 一、影像感知分析",
        "",
        f"| 指标 | 值 |",
        f"| --- | --- |",
        f"| 推断模态 | {perception.get('modality')} |",
        f"| 估计噪声σ | {perception.get('estimated_sigma')} |",
        f"| 噪声等级 | {perception.get('noise_level')} |",
        f"| 边缘密度 | {perception.get('edge_density')} |",
        f"| 纹理复杂度 | {perception.get('texture_complexity')} |",
        "",
        "## 二、处理决策轨迹",
        "",
    ]
    for entry in result["history"]:
        lines.append(f"### 第{entry['iteration']}轮")
        lines.append("")
        for i, step in enumerate(entry["plan"], 1):
            lines.append(f"{i}. `{step['tool']}` 参数: `{json.dumps(step.get('args', {}), ensure_ascii=False)}`")
        lines.append("")
    lines += ["## 三、质量评估结果", ""]
    for key, value in evaluation.items():
        lines.append(f"- **{key}**: {value}")
    lines += ["", "## 四、结论", ""]
    if "psnr" in evaluation:
        lines.append(
            f"处理后图像 PSNR 达到 **{evaluation['psnr']} dB**, "
            f"SSIM 达到 **{evaluation['ssim']}**, 去噪与结构保持效果良好。")
    else:
        lines.append("已完成无参考质量评估, 清晰度与信息熵指标见上表。")
    return "\n".join(lines)
