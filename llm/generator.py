from __future__ import annotations
from huggingface_hub import InferenceClient
from config import SETTINGS


class AnswerGenerator:
    def __init__(self) -> None:
        self.client = InferenceClient(api_key=SETTINGS.huggingface_api_key)
        self.model = SETTINGS.llm_model

    @staticmethod
    def _build_prompt(question: str, contexts: list[dict]) -> str:
        context_blocks = []
        for idx, c in enumerate(contexts, start=1):
            context_blocks.append(f"[{idx}] {c['text']} (video_id={c['video_id']})")

        joined = "\n".join(context_blocks)
        return (
            "You are a strict grounded assistant. "
            "Answer ONLY from the provided context snippets. "
            "If context is insufficient, answer exactly: "
            "I do not have enough context from the video to answer that confidently.\n\n"
            f"Question: {question}\n\n"
            f"Context snippets:\n{joined}\n\n"
            "Provide a concise, direct answer focused only on relevant facts."
        )

    def generate_answer(self, question: str, contexts: list[dict]) -> str:
        prompt = self._build_prompt(question, contexts)
        try:
            messages = [{"role": "user", "content": prompt}]
            output = self.client.chat_completion(
                messages,
                model=self.model,
                max_tokens=220,
                temperature=0.1,
            )
            return output.choices[0].message.content.strip()
        except Exception:
            if contexts:
                c = contexts[0]
                return (
                    "Based on retrieved transcript snippets: "
                    f"{c['text'][:240]}"
                )
            return "I do not have enough context from the video to answer that confidently."