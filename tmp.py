import typing as t
from enum import StrEnum

from dotenv import load_dotenv
from langfuse.openai import OpenAI
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich import print


class OptionType(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"


class CorrectOption(BaseModel):
    stem: str
    type: t.Literal[OptionType.CORRECT] = Field(
        default=OptionType.CORRECT,
        description="Indicates that this option is the correct answer.",
    )


class IncorrectOption(BaseModel):
    stem: str
    type: t.Literal[OptionType.INCORRECT] = Field(
        default=OptionType.INCORRECT,
        description="Indicates that this option is an incorrect answer.",
    )


type Option = CorrectOption | IncorrectOption


class MCQ(BaseModel):
    question: str
    options: list[Option] = Field(..., min_length=2, max_length=5)


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, env_file_encoding="utf-8", extra="ignore")
    openai_model_name: str = "gpt-4.1-nano"
    openai_api_base: str = "https://api.openai.com/v1"
    openai_api_key: SecretStr


SYSTEM_PROMPT = """You are an expert Deep Learning instructor who writes exam-quality \
multiple choice questions.

Generate a single multiple choice question about Deep Learning.

Requirements:
- The question must test conceptual understanding, not trivia.
- Provide between 2 and 5 answer options.
- Exactly one option must be marked as "correct"; all others must be marked "incorrect".
- Incorrect options must be plausible distractors, not obviously wrong.
"""


def main():
    load_dotenv(dotenv_path=".env", verbose=True, override=True)

    settings = OpenAISettings()
    client = OpenAI(
        base_url=settings.openai_api_base,
        api_key=settings.openai_api_key.get_secret_value(),
    )

    response = client.responses.parse(
        model=settings.openai_model_name,
        instructions=SYSTEM_PROMPT,
        input="Generate one multiple choice question about Deep Learning.",
        text_format=MCQ,
    )

    mcq = response.output_parsed
    print(mcq)


if __name__ == "__main__":
    main()
