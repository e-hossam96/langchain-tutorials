import typing as t
import uuid
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, InjectedToolArg, tool
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.markdown import Markdown
from sqlalchemy import select
from sqlalchemy.orm import Session
from tavily import TavilyClient

from database import DB_URI, get_session
from database import Book as BookORM
from database import User as UserORM

# observability tools


class LangFuseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    langfuse_secret_key: SecretStr
    langfuse_public_key: SecretStr
    langfuse_base_url: str


langfuse_settings = LangFuseSettings()
langfuse = Langfuse(
    public_key=langfuse_settings.langfuse_public_key.get_secret_value(),
    secret_key=langfuse_settings.langfuse_secret_key.get_secret_value(),
    base_url=langfuse_settings.langfuse_base_url,
)

# agent setup


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    openai_model_name: str = "gpt-5.6-luna"
    openai_api_base: str = "https://api.openai.com/v1"
    openai_api_key: SecretStr


llm_settings = OpenAISettings()
llm = ChatOpenAI(
    model=llm_settings.openai_model_name,
    base_url=llm_settings.openai_api_base,
    api_key=llm_settings.openai_api_key,
    use_responses_api=True,
)


@dataclass
class UserContext:
    user_name: str


class AppAgentState(t.TypedDict):
    is_authenticated_user: bool


@dataclass
class Runtime[ContextT]:
    """Per-tool-call handle for injected context/state (replaces langgraph's ToolRuntime)."""

    tool_call_id: str
    context: ContextT
    state: AppAgentState


@tool
def get_user_name_from_context(runtime: t.Annotated[Runtime[UserContext], InjectedToolArg]) -> str:
    """Retrieve the current user's name from the run context."""
    return runtime.context.user_name


@tool
def check_user_exists_in_db(user_name: str) -> bool:
    """Check whether a user with the given name exists in the database."""
    db_session: Session = get_session(DB_URI)
    result = db_session.execute(select(UserORM).where(UserORM.name == user_name))
    return result.one_or_none() is not None


@tool
def update_user_authentication_state(
    is_authenticated_user: bool, runtime: t.Annotated[Runtime[UserContext], InjectedToolArg]
) -> str:
    """Update the agent state with the user's authentication status."""
    runtime.state["is_authenticated_user"] = is_authenticated_user
    return "Successfuly updated user existing status."


@tool
def get_user_authentication_state(runtime: t.Annotated[Runtime[UserContext], InjectedToolArg]) -> dict[str, bool]:
    """Return the current authentication-related state of the agent."""
    return {"is_authenticated_user": runtime.state.get("is_authenticated_user", False)}


class WebSearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    tavily_api_key: SecretStr


web_search_settings = WebSearchSettings()
web_search_client = TavilyClient(api_key=web_search_settings.tavily_api_key.get_secret_value())


@dataclass
class WebSearchResult:
    title: str
    url: str
    content: str


@tool
def web_search(query: str, num_results: int = 5) -> list[WebSearchResult]:
    """Search the web for the given query and return the top results."""
    search_results: dict[str, t.Any] = web_search_client.search(query=query, max_results=num_results)
    results = [
        WebSearchResult(title=result["title"], url=result["url"], content=result["content"])
        for result in search_results["results"]
    ]
    return results


@dataclass
class Book:
    name: str


@tool
def get_available_books_from_db(limit: int = 10) -> list[Book]:
    """Fetch a list of available books from the database."""
    db_session: Session = get_session(DB_URI)
    result = db_session.execute(select(BookORM).limit(limit))
    return [Book(name=book_obj[0].name) for book_obj in result.all()]


@tool
def insert_books_into_db(books: list[Book]) -> bool:
    """Insert a list of new books into the database."""
    db_session: Session = get_session(DB_URI)
    try:
        for book in books:
            book_obj = BookORM(name=book.name.strip()[:200])  # constraint fom database orm
            db_session.add(book_obj)
            db_session.commit()
            db_session.refresh(book_obj)
    except Exception:
        return False

    return True


def select_tools(state: AppAgentState) -> list[BaseTool]:
    is_authenticated_user: bool = state.get("is_authenticated_user", False)
    available_tools: list[BaseTool] = [
        get_user_name_from_context,
        check_user_exists_in_db,
        update_user_authentication_state,
        get_user_authentication_state,
        web_search,
        get_available_books_from_db,
    ]
    if is_authenticated_user:
        available_tools.append(insert_books_into_db)
    return available_tools


def _accepts_runtime(tool_obj: BaseTool) -> bool:
    return "runtime" in tool_obj.args_schema.model_fields


def run_agent_turn(
    *,
    model: ChatOpenAI,
    messages: list[BaseMessage],
    state: AppAgentState,
    context: UserContext,
    config: RunnableConfig,
) -> AIMessage:
    """Run one user turn to completion: call the model, execute any tool calls it makes,
    and repeat until the model responds without further tool calls."""
    while True:
        active_tools = select_tools(state)
        tool_map = {tool_obj.name: tool_obj for tool_obj in active_tools}
        ai_message = model.bind_tools(active_tools).invoke(messages, config=config)
        messages.append(ai_message)

        if not ai_message.tool_calls:
            return ai_message

        for tool_call in ai_message.tool_calls:
            tool_obj = tool_map[tool_call["name"]]
            call_args = dict(tool_call["args"])
            if _accepts_runtime(tool_obj):
                call_args["runtime"] = Runtime(tool_call_id=tool_call["id"], context=context, state=state)
            tool_message = tool_obj.invoke({**tool_call, "args": call_args}, config=config)
            messages.append(tool_message)


SYSTEM_PROMPT = """You are a helpful AI assistant that manages a book database.

## Task

Your task is to provide help for allowed (authenticated) users to enrich the book database by \
finding a list of book names online in the requested domain.

You can also provide non-authenticated users with access to search online for books about \
a specific domain.

## User Authentication

You are provided with tools that help you authenticate the current user give their `user_name` \
provided in the context.

The Authentication Process (CRITICAL):

1. Get `user_name` from the provided context.
2. Check that user exists in the database.
3. Update the authentication flag in the state.
4. Change the allowed tools based on the authentication flag.

Perform this authentication process once at the start of the conversation. Follow the steps exactly.

## Authenticated Users

Authenticated users are allowed to search online for books, get the book list from the database, and add \
new books to the database.

## Non-Authenticated Users

Non-authenticated users get access to search online for books and get the book list from the database. \
They are not allowed to add books to the database."""


def main() -> None:
    langfuse_handler = CallbackHandler()
    rich_console = Console()

    rich_console.print(Markdown("**AI Agent is ready!**"), style="green")

    thread_id = uuid.uuid4().hex

    config: RunnableConfig = {
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_session_id": thread_id,
        },
    }

    context = UserContext(user_name="hossam")
    state: AppAgentState = {"is_authenticated_user": False}
    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT)]

    while True:
        user_input = input("User: ")
        if user_input.lower() in {"exit", "quit"}:
            rich_console.print("Exiting...", style="bold red")
            break

        messages.append(HumanMessage(content=[{"type": "input_text", "text": user_input}]))

        ai_message = run_agent_turn(
            model=llm,
            messages=messages,
            state=state,
            context=context,
            config=config,
        )

        response_content = ai_message.content[-1]["text"]
        rich_console.print("AI Agent:", style="bold green", end=" ")
        rich_console.print(Markdown(response_content), style="green")
        usage_metadata = ai_message.usage_metadata
        rich_console.print(Markdown(f"{usage_metadata}"), style="blue")
        rich_console.print()


if __name__ == "__main__":
    main()
