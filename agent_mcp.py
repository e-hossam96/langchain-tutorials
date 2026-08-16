"""Wedding planner AI Agent."""

import asyncio
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.tools import ToolRuntime, tool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
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


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, env_file_encoding="utf-8", extra="ignore")
    database_uri: SecretStr


load_dotenv(dotenv_path=".env", verbose=True, override=True)

# venue search agent

web_search_settings = WebSearchSettings()
web_search_client = TavilyClient(api_key=web_search_settings.tavily_api_key.get_secret_value())


@dataclass
class WebSearchResult:
    title: str
    url: str
    content: str


@tool
def web_search(query: str) -> list[WebSearchResult]:
    """Search web and get a summarized answer for a given query."""
    search_results = web_search_client.search(query=query, max_results=5)
    results = [
        WebSearchResult(title=result["title"], url=result["url"], content=result["content"])
        for result in search_results["results"]
    ]
    return results


venue_llm_settings = OpenAISettings()
venue_llm = ChatOpenAI(
    model=venue_llm_settings.openai_model_name,
    base_url=venue_llm_settings.openai_api_base,
    api_key=venue_llm_settings.openai_api_key,
    use_responses_api=True,
)

VENUE_SYSTEM_PROMPT = """You are a helpful venue search AI agent."""

venue_agent = create_agent(
    model=venue_llm,
    tools=[web_search],
    checkpointer=InMemorySaver(),
    system_prompt=VENUE_SYSTEM_PROMPT,
    name="Venue Search Agent",
)


# database query agent

database_llm_settings = OpenAISettings()
database_settings = DatabaseSettings()
database = SQLDatabase.from_uri(database_uri=database_settings.database_uri.get_secret_value())
database_llm = ChatOpenAI(
    model=database_llm_settings.openai_model_name,
    base_url=database_llm_settings.openai_api_base,
    api_key=database_llm_settings.openai_api_key,
    use_responses_api=True,
)
toolkit = SQLDatabaseToolkit(db=database, llm=database_llm)


DATABASE_SYSTEM_PROMPT = f"""You are a helpful database query AI agent.

You're task is to create {database.dialect} `SELECT` sql queries to answer user queries.

Limit returned results to top 10 records only."""

database_agent = create_agent(
    model=database_llm,
    tools=toolkit.get_tools(),
    checkpointer=InMemorySaver(),
    system_prompt=DATABASE_SYSTEM_PROMPT,
    name="Database Query Agent",
)


# flight search agent

flight_llm_settings = OpenAISettings()
flight_llm = ChatOpenAI(
    model=flight_llm_settings.openai_model_name,
    base_url=flight_llm_settings.openai_api_base,
    api_key=flight_llm_settings.openai_api_key,
    use_responses_api=True,
)


flight_mcp_client = MultiServerMCPClient(
    connections={"travel_server": {"transport": "streamable_http", "url": "https://mcp.kiwi.com"}}
)
flight_tools = asyncio.run(flight_mcp_client.get_tools())


@tool
def get_current_utc_time() -> str:
    """Get current utc date and time in ISO format."""
    return datetime.now(UTC).isoformat()


FLIGHT_SYSTEM_PROMPT = """You are a helpful travel assistance AI agent.

Check current date and time before making any assumptions.
Confirm flight date and time before performing any search.
Provide date in the needed format for the search flight tool (dd/mm/yyyy)."""


flight_agent = create_agent(
    model=flight_llm,
    tools=flight_tools + [get_current_utc_time],
    checkpointer=InMemorySaver(),
    system_prompt=FLIGHT_SYSTEM_PROMPT,
)


# wedding planner agent (main)


wedding_llm_settings = OpenAISettings()
wedding_llm = ChatOpenAI(
    model=wedding_llm_settings.openai_model_name,
    base_url=wedding_llm_settings.openai_api_base,
    api_key=wedding_llm_settings.openai_api_key,
    use_responses_api=True,
)


class WeddingPlannerAgentState(AgentState):
    flight_origin: str
    flight_destination: str
    flight_date: str
    venue_guest_count: int
    music_genre: str


@tool
def update_agent_state(
    flight_origin: str,
    flight_destination: str,
    flight_date: str,
    venue_guest_count: str,
    music_genre: str,
    runtime: ToolRuntime,
) -> Command:
    """Update the state when you know all of the values."""
    return Command(
        update={
            "flight_origin": flight_origin,
            "flight_destination": flight_destination,
            "flight_date": flight_date,
            "venue_guest_count": venue_guest_count,
            "music_genre": music_genre,
            "messages": [
                ToolMessage(
                    "Successfully updated wedding agent state",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
async def search_flight(flight_origin: str, flight_destination: str, flight_date: str) -> str:
    """Travel agent searches for flights to the desired destination wedding location."""
    user_prompt = f"""Find a comfort flight from {flight_origin} to {flight_destination} \
on {flight_date} for a recently married couple."""
    response = await flight_agent.ainvoke(
        input={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt},
                    ],
                },
            ]
        },
    )
    return response["messages"][-1].content[-1]["text"]


@tool
def search_venue(
    flight_destination: str,
    flight_date: str,
    venue_guest_count: int,
) -> str:
    """Venue agent chooses the best venue for the given location and capacity."""
    user_prompt = f"""Find a decent wedding venue in {flight_destination} \
for a recently married couple who will arrive in {flight_date} that can host {venue_guest_count} people. \
Venue needs to be available anytime after {flight_date}."""
    response = venue_agent.invoke(
        input={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt},
                    ],
                },
            ]
        },
    )
    return response["messages"][-1].content[-1]["text"]


@tool
def find_music_list(
    music_genre: str,
) -> str:
    """Playlist agent curates the perfect playlist for the given genre."""
    user_prompt = f"""Find a decent wedding {music_genre} music list for a recently married couple."""
    response = database_agent.invoke(
        input={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt},
                    ],
                },
            ]
        },
    )
    return response["messages"][-1].content[-1]["text"]


WEDDING_SYSTEM_PROMPT = """You are a wedding coordinator. Delegate tasks to your \
specialists for flights, venues and playlists.
First find all the information you need to update the state. Once that \
is done you can delegate the tasks."""


wedding_agent = create_agent(
    model=wedding_llm,
    tools=[update_agent_state, search_flight, search_venue, find_music_list],
    checkpointer=InMemorySaver(),
    system_prompt=WEDDING_SYSTEM_PROMPT,
    state_schema=WeddingPlannerAgentState,
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

            response = await wedding_agent.ainvoke(
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

            answer = response["messages"][-1].content[-1]["text"]
            rich_console.print("AI Agent:", style="bold white", end=" ")
            rich_console.print(Markdown(answer), style="white")
            rich_console.print()

        except KeyboardInterrupt:
            rich_console.print("Exiting...", style="bold red")
            break


if __name__ == "__main__":
    asyncio.run(main())
