"""协调Agent: 编排 感知→决策→执行→评估 闭环, 支持反思式迭代优化."""

import numpy as np

from ..tools.registry import ToolExecutor
from .memory import CaseMemory
from .skills import SkillLoader
from .subagents import (EvaluationAgent, ExecutionAgent, PerceptionAgent,
                        PlanningAgent)


class Coordinator:
    def __init__(self, max_iterations: int = 2, verbose: bool = True):
        self.memory = CaseMemory()
        self.skills = SkillLoader()
        self.perception_agent = PerceptionAgent()
        self.planning_agent = PlanningAgent(self.memory, self.skills)
        self.execution_agent = ExecutionAgent()
        self.evaluation_agent = EvaluationAgent()
        self.max_iterations = max_iterations
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def process(self, image: np.ndarray, reference: np.ndarray | None = None,
                filename: str = "") -> dict:
        """处理一张医学影像, 返回结果图像、评估指标与完整决策轨迹."""
        executor = ToolExecutor(image, reference, filename)

        self._log("[感知Agent] 分析影像特征...")
        perception = self.perception_agent.run(image, filename)
        self._log(f"  -> {perception}")

        self._log("[决策Agent] 生成处理计划...")
        plan, planner_type = self.planning_agent.run(perception)
        self._log(f"  -> 规划器: {planner_type}, 计划: {plan}")

        evaluation: dict = {}
        history: list[dict] = []
        for iteration in range(1, self.max_iterations + 1):
            self._log(f"[执行Agent] 第{iteration}轮执行 ({len(plan)}步)...")
            self.execution_agent.run(executor, plan)

            evaluation = self.evaluation_agent.run(executor)
            self._log(f"[评估Agent] 评估结果: {evaluation}")
            history.append({"iteration": iteration, "plan": plan,
                            "evaluation": evaluation})

            if self.evaluation_agent.is_satisfactory(evaluation):
                self._log("[评估Agent] 质量达标, 结束迭代")
                break
            if iteration < self.max_iterations:
                self._log("[评估Agent] 质量未达标, 触发反思重规划")
                plan = self.evaluation_agent.suggest_refinement(evaluation, plan)
                executor.current = executor.original.copy()  # 从原始噪声图重新处理
                executor.denoised = None

        self.memory.record(perception, history[0]["plan"], evaluation)
        return {
            "output": executor.current,
            "perception": perception,
            "planner": planner_type,
            "history": history,
            "evaluation": evaluation,
            "trace": executor.trace,
        }
