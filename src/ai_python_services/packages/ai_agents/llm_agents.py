from abc import ABC, abstractmethod
from json import loads as json_loads
from os import getenv
from typing import Any

from anthropic import Anthropic
from google import genai
from langchain_core.output_parsers import JsonOutputParser
from openai import OpenAI
from pydantic import BaseModel

from .ai_enum import (
    AnthropicModel,
    GoogleModel,
    Language,
    OpenAIModel,
)


class LLMAgent(ABC):
    """Base interface for all LLM-based agents."""

    @abstractmethod
    def generate(
        self,
        prompt: str = "You are an AI agent named Felix.",
        input_text: str = "What is your name?",
        output_format: type[BaseModel] | None = None,
    ) -> str | dict[str, Any]:
        """Generate a response from the LLM.

        Args:
            prompt: The input prompt for the LLM
            input_text: Additional input text to append to prompt
            output_format: Expected output format (BaseModel for JSON, None for text)

        Returns:
            The generated response as string or parsed JSON
        """
        pass


class OpenAIAgent(LLMAgent):
    """OpenAI-specific LLM agent."""

    def __init__(
        self,
        model_name: OpenAIModel = OpenAIModel.GPT_4O_MINI,
        language: Language = Language.VI_VN,
    ):
        self.model_name = model_name
        self.language = language

        api_key = getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "API key must be set in the environment variable 'OPENAI_API_KEY'"
            )
        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        prompt: str = "You are an AI agent named Felix.",
        input_text: str = "What is your name?",
        output_format: type[BaseModel] | None = None,
    ) -> str | dict[str, Any]:
        """Generate a response using OpenAI's LLM."""

        full_prompt = f"{prompt}\n\nUser: {input_text}\n\nPlease respond in {self.language.toText()}."

        if output_format is None:
            response = self.client.chat.completions.create(
                model=self.model_name.value,
                messages=[
                    {"role": "system", "content": full_prompt},
                    {
                        "role": "user",
                        "content": f"{input_text}",
                    },
                ],
                temperature=0,
            )
            return response.choices[0].message.content or ""
        else:
            response = self.client.beta.chat.completions.parse(
                model=self.model_name.value,
                messages=[
                    {
                        "role": "system",
                        "content": full_prompt,
                    },
                    {"role": "user", "content": input_text},
                ],
                response_format=output_format,
                temperature=0,
            )
            return (
                response.choices[0].message.parsed.model_dump()
                if response.choices[0].message.parsed
                else {}
            )


class AnthropicAgent(LLMAgent):
    """Anthropic-specific LLM agent."""

    def __init__(
        self,
        model_name: AnthropicModel = AnthropicModel.CLAUDE_3_5_HAIKU,
        language: Language = Language.VI_VN,
    ):
        self.model_name = model_name
        self.language = language

        api_key = getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "API key must be set in the environment variable 'ANTHROPIC_API_KEY'"
            )
        self.client = Anthropic(api_key=api_key)

    def generate(
        self,
        prompt: str = "You are an AI agent named Felix.",
        input_text: str = "What is your name?",
        output_format: type[BaseModel] | None = None,
    ) -> str | dict[str, Any]:
        """Generate a response using Anthropic's Claude."""

        if output_format is None:
            response = self.client.messages.create(
                model=self.model_name.value,
                max_tokens=1024,
                system=f"{prompt}\n\nPlease respond in {self.language.toText()}.",
                messages=[{"role": "user", "content": input_text}],
                temperature=0,
            )
            return response.content[0].text  # type: ignore
        else:
            # JSON output with format instructions
            parser = JsonOutputParser(pydantic_object=output_format)
            format_instructions = parser.get_format_instructions()

            response = self.client.messages.create(
                model=self.model_name.value,
                max_tokens=1024,
                system=f"{prompt}\n{format_instructions}\n\nPlease respond in {self.language.toText()}.",
                messages=[{"role": "user", "content": input_text}],
                temperature=0,
            )

            response_text = response.content[0].text  # type: ignore
            return parser.parse(response_text)  # type: ignore


class GoogleAgent(LLMAgent):
    """Google-specific LLM agent."""

    def __init__(
        self,
        model_name: GoogleModel = GoogleModel.GEMINI_2_0_FLASH,
        language: Language = Language.VI_VN,
    ):
        self.model_name = model_name
        self.language = language

        api_key = getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "API key must be set in the environment variable 'GOOGLE_API_KEY'"
            )
        self.client = genai.Client(api_key=api_key)

    def generate(
        self,
        prompt: str = "You are an AI agent named Felix.",
        input_text: str = "What is your name?",
        output_format: type[BaseModel] | None = None,
    ) -> str | dict[str, Any]:
        """Generate a response using Google's Gemini."""

        full_content = f"{prompt}\n\nUser: {input_text}\n\nPlease respond in {self.language.toText()}."

        if output_format is None:
            response = self.client.models.generate_content(
                model=self.model_name.value,
                contents=[full_content],
            )
            return response.text if response.text else ""
        else:
            response = self.client.models.generate_content(
                model=self.model_name.value,
                contents=[full_content],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": output_format,
                },
            )
            response_text = response.text if response.text else "{}"
            return json_loads(response_text)


# Test example
if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv

    load_dotenv()

    class TaskResult(BaseModel):
        """Example custom output format for task results."""

        name: str

    # Test OpenAI
    print("Testing OpenAI Agent...")
    agent = OpenAIAgent(model_name=OpenAIModel.GPT_4O_MINI)
    response = agent.generate(
        prompt="You are an AI assistant named Felix.",
        input_text="Hello, what's your name?",
        output_format=TaskResult,
    )
    print(f"OpenAI JSON output: {response}")
    print(f"Type: {type(response)}")
    assert isinstance(response, dict), "OpenAI JSON test failed"

    # Test string output
    response = agent.generate(
        prompt="You are an AI assistant named Felix.",
        input_text="Hello, what's your name?",
    )
    print(f"OpenAI text output: {response}")
    print(f"Type: {type(response)}")
    assert isinstance(response, str), "OpenAI text test failed"

    # Test Anthropic
    print("\nTesting Anthropic Agent...")
    agent = AnthropicAgent(model_name=AnthropicModel.CLAUDE_3_5_HAIKU)
    response = agent.generate(
        prompt="You are an AI assistant named Felix.",
        input_text="Hello, what's your name?",
        output_format=TaskResult,
    )
    print(f"Anthropic JSON output: {response}")
    print(f"Type: {type(response)}")
    assert isinstance(response, dict), "Anthropic JSON test failed"

    # Test Google
    print("\nTesting Google Agent...")
    agent = GoogleAgent(model_name=GoogleModel.GEMINI_2_0_FLASH)
    response = agent.generate(
        prompt="You are an AI assistant named Felix.",
        input_text="Hello, what's your name?",
        output_format=TaskResult,
    )
    print(f"Google JSON output: {response}")
    print(f"Type: {type(response)}")
    assert isinstance(response, dict), "Google JSON test failed"

    # Test string output
    agent = GoogleAgent(model_name=GoogleModel.GEMINI_2_0_FLASH)
    response = agent.generate(
        prompt="You are an AI assistant named Felix.",
        input_text="Hello, what's your name?",
    )
    print(f"Google text output: {response}")
    print(f"Type: {type(response)}")
    assert isinstance(response, str), "Google text test failed"

    print("\nAll tests for LLM agents passed successfully!\n")
