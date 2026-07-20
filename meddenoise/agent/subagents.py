"""子Agent定义: 感知 / 决策 / 执行 / 评估 四个专职子Agent."""

import numpy as np

from ..tools import analysis, metrics
from ..tools.registry import ToolExecutor
from .memory import CaseMemory
from .planner import make_plan
from .skills import SkillLoader


class PerceptionAgent:
    """感知Agent: 分析输入影像的噪声特性、结构特征与模态."""

    def run(self, image: np.ndarray, filename: str = "") -> dict:
        return analysis.analyze_image(image, filename)


class PlanningAgent:
    """决策Agent: 结合领域技能与历史案例记忆, 生成处理计划."""

    def __init__(self, memory: CaseMemory, skills: SkillLoader):
        self.memory = memory
        self.skills = skills

    def run(self, perception: dict) -> tuple[list[dict], str]:
        similar = self.memory.find_similar(perception)
        skills_context = self.skills.get("denoise-strategy")
        return make_plan(perception, similar, skills_context)


class ExecutionAgent:
    """执行Agent: 按计划依次调用工具, 维护图像状态与执行轨迹."""

    def run(self, executor: ToolExecutor, plan: list[dict]) -> list[dict]:
        for step in plan:
            executor.execute(step["tool"], step.get("args", {}))
        return executor.trace


class EvaluationAgent:
    """评估Agent: 质量评估与反思, 决定是否需要重新规划(迭代优化)."""

    def __init__(self, target_psnr: float = 28.0, target_ssim: float = 0.80):
        self.target_psnr = target_psnr
        self.target_ssim = target_ssim

    def run(self, executor: ToolExecutor) -> dict:
        result = metrics.no_reference_metrics(executor.current)
        if executor.reference is not None:
            target = (executor.denoised if executor.denoised is not None
                      else executor.current)
            result.update(metrics.full_reference_metrics(executor.reference,
                                                         target))
        return result

    def is_satisfactory(self, evaluation: dict) -> bool:
        if "psnr" not in evaluation:
            return True  # 无参考图时不迭代
        return (evaluation["psnr"] >= self.target_psnr
                and evaluation["ssim"] >= self.target_ssim)

    def suggest_refinement(self, evaluation: dict, last_plan: list[dict]) -> list[dict]:
        """反思: 若质量不达标, 改用更强的结构感知去噪方案."""
        used, last_sigma = set(), None
        for step in last_plan:
            if step["tool"] == "denoise_image":
                used.add(step["args"].get("method"))
                last_sigma = step["args"].get("sigma", last_sigma)
        method = "sa-bm3d" if "sa-bm3d" not in used else "bm3d"
        args: dict = {"method": method}
        if last_sigma:
            args["sigma"] = round(last_sigma * 1.3, 2)  # 残留噪声 -> 提升去噪强度
        return [
            {"tool": "denoise_image", "args": args},
            {"tool": "enhance_image", "args": {"method": "unsharp"}},
            {"tool": "evaluate_quality", "args": {}},
        ]
