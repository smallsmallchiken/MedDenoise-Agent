"""案例记忆库: 记录历史处理案例, 支持相似案例检索以复用成功策略."""

import json
from pathlib import Path

MEMORY_FILE = Path(__file__).resolve().parents[2] / "memory" / "cases.jsonl"


class CaseMemory:
    def __init__(self, memory_file: Path = MEMORY_FILE):
        self.memory_file = memory_file
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.cases = self._load()

    def _load(self) -> list[dict]:
        if not self.memory_file.exists():
            return []
        with open(self.memory_file, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def record(self, perception: dict, plan: list[dict], evaluation: dict) -> None:
        case = {"perception": perception, "plan": plan, "evaluation": evaluation}
        self.cases.append(case)
        with open(self.memory_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    def find_similar(self, perception: dict, top_k: int = 3) -> list[dict]:
        """按噪声sigma与边缘密度的欧氏距离检索最相似历史案例."""

        def distance(case: dict) -> float:
            p = case["perception"]
            d_sigma = (p.get("estimated_sigma", 0) - perception.get("estimated_sigma", 0)) / 50.0
            d_edge = p.get("edge_density", 0) - perception.get("edge_density", 0)
            return d_sigma ** 2 + d_edge ** 2

        return sorted(self.cases, key=distance)[:top_k]
