import math
import ollama
from collections.abc import Sequence


class Observer:
    task_spec: str
    task_embedding: Sequence[float]

    def __init__(self, task_spec: str):
        self.task_spec = task_spec
        self.task_embedding = ollama.embed(
            model="nomic-embed-text:latest", input=task_spec
        ).embeddings[0]
        self._history: list[dict] = []

    def observe(self, node_name: str, state: dict) -> dict:
        framing = self._extract_framing(state)
        drift = (
            self._cosine_similarity(
                self.task_embedding,
                ollama.embed(model="nomic-embed-text:latest", input=framing).embeddings[
                    0
                ],
            )
            if framing
            else None
        )

        tokens = self._count_tokens(state)
        system_prompt = state.get("system_prompt", "")
        reasoning = state.get("reasoning", "")

        report = {
            "node": node_name,
            "drift_score": drift,
            "context_tokens": tokens,
            "system_prompt": system_prompt,
            "reasoning": reasoning,
        }
        self._history.append(report)
        return report

    def get_report(self, node_name: str, state: dict) -> dict:
        return self.observe(node_name, state)

    def drift_scores(self) -> list[float]:
        return [r["drift_score"] for r in self._history if r["drift_score"] is not None]

    def peak_context_tokens(self) -> int | None:
        counts = [
            r["context_tokens"]
            for r in self._history
            if r["context_tokens"] is not None
        ]
        return max(counts) if counts else None

    def _extract_framing(self, state: dict) -> str:
        messages = state.get("messages", [])
        if not messages:
            return state.get("task", "")
        last = messages[-1]
        if isinstance(last, dict):
            return last.get("content", "")
        return str(last)

    def _count_tokens(self, state: dict) -> int:
        messages = state.get("messages", [])
        total = 0
        for msg in messages:
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            total += len(content) // 4
        return total

    @staticmethod
    def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)
