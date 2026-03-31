from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from config import SETTINGS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

SYSTEM_PROMPT = (
    "You are a strict grounded assistant. "
    "Answer ONLY from the provided context snippets. "
    "If context is insufficient, answer exactly: "
    "I do not have enough context from the video to answer that confidently.\n\n"
    "Question: {question}\n\n"
    "Context snippets:\n{context}\n\n"
    "Provide a concise, direct answer focused only on relevant facts."
)

class BaseAnswerGenerator(ABC):
    def __init__(self) -> None:
        self.prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
        self.parser = StrOutputParser()

    def _build_context(self, contexts: list[dict]) -> str:
        blocks = []
        for idx, item in enumerate(contexts, start=1):
            blocks.append(f"[{idx}] {item['text']} (video_idx-{item['video_id']})")
        return "\n".join(blocks)
    
    def _fallback_answer(self, contexts: list[dict]) -> str:
        if contexts:
            first = contexts[0]
            return f"Based on retrieved snippets: {first['text'][:240]}"
        return "I do not have enough context form the cideo to answer that confidently."
    
    @abstractmethod
    def generate_answer(self, question:str, contexts: list[dict]) -> str:
        raise NotImplementedError
    
class HuggingFaceAnswerGenerator(BaseAnswerGenerator):
    def __init__(self) -> None:
        super().__init__()
        llm = HuggingFaceEndpoint(
            repo_id=SETTINGS.llm_model,
            task='text-generation',
            huggingfacehub_api_token = SETTINGS.huggingface_api_key or None,
            max_new_tokens=220,
            temperature=0.1,
            do_sample=False
        )
        self.chat_model = ChatHuggingFace(llm=llm)

    def generate_answer(self, question: str, contexts: list[dict]) -> str:
        context_text = self._build_context(contexts)
        chain = self.prompt | self.chat_model | self.parser
        try:
            return chain.invoke({"question": question, "context": context_text}).strip()
        except Exception:
            return self._fallback_answer(contexts)
        
class GroqAnswerGenerator(BaseAnswerGenerator):
    def __init__(self) -> None:
        super().__init__()
        try:
            from groq import Groq
        except Exception as e:
            raise RuntimeError(
                "Groq provider requested but groq package is not installed.\nInstall it with: pip install groq"
            ) from e
        
        if not SETTINGS.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        
        configured_model = getattr(SETTINGS, "groq_llm_model", None)
        self.model_name = configured_model
        self.client = Groq(api_key=SETTINGS.groq_api_key)

    def generate_answer(self, question:str, contexts: list[dict]) -> str:
        context_text = self._build_context(contexts) #what was the existing context like that we had to pass it through a custom _build_context layer what does it do differently that helped us. We could have kept it as it is... why did we do it?
        user_prompt = (
            f"Question: {question}\n\n"
            f"Context Snippets: {context_text}\n\n"
            "Answer only from context"
        )

        try:
            response = self.client.chat.completions.create(
                model = self.model_name,
                temperature=0.1,
                max_tokens=220,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.replace("{question}", "").replace("{context}", "")},
                    {"role": "user", "content": user_prompt}
                ]#tf is this message block
            )
            text = (response.choices[0].message.content or "").strip()
            if not text:
                return self._fallback_answer(contexts)
            return text
        except Exception:
            return self._fallback_answer(contexts)
        
def create_answer_generator(provider: str | None = None) -> BaseAnswerGenerator:
    selected = (provider or SETTINGS.llm_provider or "huggingface").strip().lower()

    if selected == "groq":
        return GroqAnswerGenerator()
    
    if selected == "huggingface":
        return HuggingFaceAnswerGenerator()
    
    raise ValueError(f"Unsupported LLM provider: {selected}")
