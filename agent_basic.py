"""
This is a simple AI agent using langchain.

Agent Features:
    - Short term memory
    - Web search tool
    - Local model using vLLM
"""

import uuid
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich import print
from rich.console import Console
from rich.markdown import Markdown
from tavily import TavilyClient


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, env_file_encoding="utf-8", extra="ignore")
    openai_model_name: str = "gpt-4.1-nano"
    openai_api_base: str = "https://api.openai.com/v1"
    openai_api_key: SecretStr


class WebSearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, env_file_encoding="utf-8", extra="ignore")
    tavily_api_key: SecretStr


@lru_cache(maxsize=8)
def get_web_search_settings() -> WebSearchSettings:
    return WebSearchSettings()


@lru_cache(maxsize=8)
def get_tavily_client() -> TavilyClient:
    settings = get_web_search_settings()
    return TavilyClient(api_key=settings.tavily_api_key.get_secret_value())


@dataclass
class WebSearchResult:
    title: str
    url: str
    content: str


@tool
def web_search(query: str) -> list[WebSearchResult]:
    """Search web and get a summarized answer for a given query."""
    web_search_client = get_tavily_client()
    search_results = web_search_client.search(query=query, include_answer=False, max_results=5)
    results = [
        WebSearchResult(title=result["title"], url=result["url"], content=result["content"])
        for result in search_results["results"]
    ]
    print(results)
    return results


SYSTEM_PROMPT = """You are a helpful AI agent with access to a web search tool.

You should use the web search tool to answer user queries that require up-to-date information from the web.

Whenever user asks for current or recent info or data, \
use the web search tool to get current date and time first before searching for the info or the data.
"""


def main():
    load_dotenv(dotenv_path=".env", verbose=True, override=True)

    openai_settings = OpenAISettings()

    langfuse_handler = CallbackHandler()
    rich_console = Console()

    model = ChatOpenAI(
        model=openai_settings.openai_model_name,
        base_url=openai_settings.openai_api_base,
        api_key=openai_settings.openai_api_key,
        use_responses_api=True,
    )

    agent = create_agent(
        model=model,
        tools=[web_search],
        checkpointer=InMemorySaver(),
        system_prompt=SYSTEM_PROMPT,
    )

    rich_console.print(Markdown("**AI Agent is ready!**"), style="green")

    config = {
        "configurable": {"thread_id": uuid.uuid5(namespace=uuid.NAMESPACE_DNS, name="langchain").hex},
        "callbacks": [langfuse_handler],
    }
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in {"exit", "quit"}:
                rich_console.print("Exiting...", style="bold red")
                break

            response = agent.invoke(
                input={
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input,
                        },
                    ]
                },
                config=config,
                stream_mode="values",
            )
            print(response)

            answer = response["messages"][-1].content[-1]["text"]

            rich_console.print("AI Agent:", style="bold white", end=" ")
            rich_console.print(Markdown(answer), style="white")
            rich_console.print()

        except KeyboardInterrupt:
            rich_console.print("Exiting...", style="bold red")
            break


if __name__ == "__main__":
    main()
