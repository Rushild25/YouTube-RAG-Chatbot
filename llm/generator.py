from __future__ import annotations
from config import SETTINGS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


class AnswerGenerator:
    def __init__(self) -> None:
        llm=HuggingFaceEndpoint(
            repo_id=SETTINGS.llm_model,
            model=SETTINGS.llm_model,
            task="text-generation",
            huggingfacehub_api_token=SETTINGS.huggingface_api_key or None,
            max_new_tokens=220,
            temperature=0.1,
            do_sample=False,
        )
        self.chat_model = ChatHuggingFace(llm=llm)
        self.prompt = ChatPromptTemplate.from_template(
            "You are a strict grounded assistant. "
            "Answer ONLY from the provided context snippets. "
            "If context is insufficient, answer exactly: "
            "I do not have enough context from the video to answer that confidently.\n\n"
            "Question: {question}\n\n"
            "Context snippets:\n{context}\n\n"
            "Provide a concise, direct answer focused only on relevant facts."
        )
        self.parser = StrOutputParser()

    def _build_context(self, contexts: list[dict]) -> str:
        context_blocks = []
        for idx, c in enumerate(contexts, start=1):
            context_blocks.append(f"[{idx}] {c['text']} (video_id={c['video_id']})")
        return "\n".join(context_blocks)

    def generate_answer(self, question: str, contexts: list[dict]) -> str:
        context_text = self._build_context(contexts)
        chain = self.prompt | self.chat_model | self.parser
        try:
            return chain.invoke({"question": question, "context": context_text}).strip()
        except Exception:
            if contexts:
                c = contexts[0]
                return (
                    "Based on retrieved transcript snippets: "
                    f"{c['text'][:240]}"
                )
            return "I do not have enough context from the video to answer that confidently."