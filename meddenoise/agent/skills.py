"""技能加载器: 从 skills/ 目录加载 SKILL.md 领域知识, 注入Agent上下文."""

import re
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills"


class SkillLoader:
    def __init__(self, skills_dir: Path = SKILLS_DIR):
        self.skills: dict[str, dict] = {}
        if skills_dir.exists():
            for f in sorted(skills_dir.rglob("SKILL.md")):
                meta, body = self._parse(f.read_text(encoding="utf-8"))
                name = meta.get("name", f.parent.name)
                self.skills[name] = {"meta": meta, "body": body}

    @staticmethod
    def _parse(text: str) -> tuple[dict, str]:
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text
        meta = {}
        for line in match.group(1).splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip()
        return meta, match.group(2).strip()

    def descriptions(self) -> str:
        return "\n".join(
            f"- {name}: {s['meta'].get('description', '')}" for name, s in self.skills.items()
        ) or "(无可用技能)"

    def get(self, name: str) -> str:
        skill = self.skills.get(name)
        return skill["body"] if skill else f"未知技能: {name}"
