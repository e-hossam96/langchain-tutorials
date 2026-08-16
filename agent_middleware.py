"""Agent Middleware for messages summarization and trimming."""

import asyncio
import uuid
from datetime import UTC, datetime

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich import print
from rich.console import Console
from rich.markdown import Markdown


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, env_file_encoding="utf-8", extra="ignore")
    openai_model_name: str = "gpt-4.1-nano"
    openai_api_base: str = "https://api.openai.com/v1"
    openai_api_key: SecretStr


load_dotenv(dotenv_path=".env", verbose=True, override=True)


llm_settings = OpenAISettings()
llm = ChatOpenAI(
    model=llm_settings.openai_model_name,
    base_url=llm_settings.openai_api_base,
    api_key=llm_settings.openai_api_key,
    use_responses_api=True,
)


@tool
def get_current_utc_time() -> str:
    """Get current utc date and time in ISO format."""
    return datetime.now(UTC).isoformat()


SYSTEM_PROMPT = """You are a helpful AI assistant."""

summarization_middleware = SummarizationMiddleware(
    model=llm,
    keep=("messages", 4),
    trigger=[("tokens", 4096), ("messages", 12)],
)


agent = create_agent(
    model=llm,
    tools=[get_current_utc_time],
    checkpointer=InMemorySaver(),
    system_prompt=SYSTEM_PROMPT,
    name="AI Agent",
    middleware=[summarization_middleware],
)


async def main():
    langfuse_handler = CallbackHandler()
    rich_console = Console()

    rich_console.print(Markdown("**AI Agent is ready!**"), style="green")

    config = {
        "configurable": {
            "thread_id": uuid.uuid5(namespace=uuid.NAMESPACE_DNS, name="langchain").hex,
        },
        "callbacks": [langfuse_handler],
    }

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in {"exit", "quit"}:
                rich_console.print("Exiting...", style="bold red")
                break

            response = await agent.ainvoke(
                input={
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": user_input},
                            ],
                        },
                    ]
                },
                config=config,
                stream_mode="values",
            )
            print(response)

            response_content = response["messages"][-1].content[-1]["text"]
            rich_console.print("AI Agent:", style="bold blue", end=" ")
            rich_console.print(Markdown(response_content), style="blue")
            rich_console.print()

        except (KeyboardInterrupt, EOFError, asyncio.CancelledError):
            break


if __name__ == "__main__":
    rich_console = Console()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        rich_console.print("Exiting...", style="bold red")
