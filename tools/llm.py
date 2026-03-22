import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.llm = ChatOpenAI(
            api_key=api_key or settings.openai_api_key,
            model=model or settings.openai_model_smart,
            temperature=0.3,
        )

    async def complete(
        self,
        system: str,
        user: str,
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.3,
    ) -> str | BaseModel:
        llm = self.llm.with_config({"temperature": temperature})

        if response_model is not None:
            structured_llm = llm.with_structured_output(response_model)
            result = await structured_llm.ainvoke([
                SystemMessage(content=system),
                HumanMessage(content=user),
            ])
            return result

        response = await llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=user),
        ])
        return response.content

    async def complete_raw(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        llm = self.llm.with_config({
            "temperature": temperature,
            "max_tokens": max_tokens,
        })

        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            else:
                langchain_messages.append(HumanMessage(content=msg["content"]))

        response = await llm.ainvoke(langchain_messages)
        return response.content
