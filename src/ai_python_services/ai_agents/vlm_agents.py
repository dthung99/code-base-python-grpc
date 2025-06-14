from os import getenv
from base64 import b64encode
from typing import Any
from abc import ABC, abstractmethod

from langchain_core.output_parsers import JsonOutputParser

from pydantic import BaseModel
from json import loads as json_loads

from openai import OpenAI
from google import genai
from google.genai import types
from anthropic import Anthropic

from ai_python_services.ai_agents.ai_enum import (
    AnthropicVisionModel,
    GoogleVisionModel,
    Language,
    OpenAIVisionModel,
)


class VLMAgent(ABC):
    """Base interface for all Vision-Language Model agents."""

    @abstractmethod
    def analyze_images(
        self,
        images: list[bytes],
        input_text: str = "",
        output_format: type[BaseModel] | None = None,
        images_mime_type: list[str] | str = "image/png",
    ) -> str | dict[str, Any]:
        """Analyze an image with a text prompt.

        Args:
            image_input: Image file path, URL, bytes, or PIL Image object
            prompt: The text prompt for analyzing the image
            output_format: Expected output format (BaseModel for JSON, None for text)

        Returns:
            The analysis response as string or parsed JSON
        """
        pass


class OpenAIVisionAgent(VLMAgent):
    """OpenAI-specific Vision-Language Model agent."""

    def __init__(
        self,
        model_name: OpenAIVisionModel = OpenAIVisionModel.GPT_4_1,
        prompt: str = "",
        language: Language = Language.VI_VN,
    ):
        self.model_name = model_name
        self.prompt = prompt
        self.language = language

        api_key = getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "API key must be set in the environment variable 'OPENAI_API_KEY'"
            )
        self.client = OpenAI(api_key=api_key)

    def analyze_images(
        self,
        images: list[bytes],
        input_text: str = "",
        output_format: type[BaseModel] | None = None,
        images_mime_type: list[str] | str = "image/png",
    ) -> str | dict[str, Any]:
        """Analyze an image using OpenAI's vision model."""
        image_inputs = []
        if len(images) >= 1:
            image_inputs = [b64encode(image).decode("utf-8") for image in images]

        content: list[Any] = []
        if isinstance(images_mime_type, str):
            images_mime_types = [images_mime_type] * len(image_inputs)
        else:
            assert len(images_mime_type) == len(image_inputs)
            images_mime_types = images_mime_type
        for i, image in enumerate(image_inputs):
            content.append(
                {
                    "type": "input_text",
                    "text": f"Image {i + 1}",
                }
            )
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{images_mime_types[i]};base64,{image}",
                },
            )

        if output_format is None:
            response = self.client.responses.parse(
                model=self.model_name.value,
                input=[
                    {
                        "role": "developer",
                        "content": self.prompt
                        + f"\n{input_text}"
                        + f"\nPlease respond in {self.language.toText()}",
                    },
                    {
                        "role": "user",
                        "content": content,
                    },
                ],
            )
            return response.output_text
        else:
            response = self.client.responses.parse(
                model=self.model_name.value,
                input=[
                    {
                        "role": "developer",
                        "content": self.prompt
                        + f"\n{input_text}"
                        + f"\nPlease respond in {self.language.toText()}\n",
                    },
                    {
                        "role": "user",
                        "content": content,
                    },
                ],
                text_format=output_format,
            )
            return json_loads(response.output_text)


class AnthropicVisionAgent(VLMAgent):
    """Anthropic-specific Vision-Language Model agent."""

    def __init__(
        self,
        model_name: AnthropicVisionModel = AnthropicVisionModel.CLAUDE_OPUS_4,
        prompt: str = "",
        language: Language = Language.VI_VN,
    ):
        self.model_name = model_name
        self.prompt = prompt
        self.language = language

        api_key = getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "API key must be set in the environment variable 'ANTHROPIC_API_KEY'"
            )
        self.client = Anthropic(api_key=api_key)

    def analyze_images(
        self,
        images: list[bytes],
        input_text: str = "",
        output_format: type[BaseModel] | None = None,
        images_mime_type: list[str] | str = "image/png",
    ) -> str | dict[str, Any]:
        """Analyze an image using Anthropic's vision model."""

        # Prepare content blocks
        content = []

        # Add the text prompt first
        content.append(
            {
                "type": "text",
                "text": self.prompt
                + f"\n{input_text}"
                + f"\nPlease respond in {self.language.toText()}",
            }
        )

        # Handle images MIME types
        if isinstance(images_mime_type, str):
            images_mime_types = [images_mime_type] * len(images)
        else:
            assert len(images_mime_type) == len(images)
            images_mime_types = images_mime_type
        # Add each image
        for i, image_bytes in enumerate(images):
            base64_image = b64encode(image_bytes).decode("utf-8")
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": images_mime_types[i],
                        "data": base64_image,
                    },
                }
            )

        # Build the message
        messages = [{"role": "user", "content": content}]

        if output_format is None:
            # String output
            response = self.client.messages.create(
                model=self.model_name.value,
                max_tokens=1024,
                messages=messages,  # type: ignore
            )
            return response.content[0].text  # type: ignore
        else:
            # JSON output with format instructions
            parser = JsonOutputParser(pydantic_object=output_format)
            format_instructions = parser.get_format_instructions()

            # Update prompt to include format instructions
            content[0]["text"] = f"{content[0]['text']}\n{format_instructions}\n"

            response = self.client.messages.create(
                model=self.model_name.value,
                max_tokens=1024,
                messages=messages,  # type: ignore
            )

            # Parse the JSON response
            response_text = response.content[0].text  # type: ignore
            return parser.parse(response_text)  # type: ignore


class GoogleVisionAgent(VLMAgent):
    """Google-specific Vision-Language Model agent."""

    def __init__(
        self,
        model_name: GoogleVisionModel = GoogleVisionModel.GEMINI_2_0_FLASH,
        prompt: str = "",
        language: Language = Language.VI_VN,
    ):
        self.model_name = model_name
        self.prompt = prompt
        self.language = language

        api_key = getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "API key must be set in the environment variable 'GOOGLE_API_KEY'"
            )
        self.client = genai.Client(api_key=api_key)

    def analyze_images(
        self,
        images: list[bytes],
        input_text: str = "",
        output_format: type[BaseModel] | None = None,
        images_mime_type: list[str] | str = "image/png",
    ) -> str | dict[str, Any]:
        """Analyze an image using Google's Gemini vision model."""

        # Handle images MIME types
        if isinstance(images_mime_type, str):
            images_mime_types = [images_mime_type] * len(images)
        else:
            assert len(images_mime_type) == len(images)
            images_mime_types = images_mime_type

        # Prepare content - start with prompt
        contents: list[Any] = [
            self.prompt
            + f"\n{input_text}"
            + f"\nPlease respond in {self.language.toText()}",
        ]

        # Add each image using types.Part.from_bytes
        for i, image_bytes in enumerate(images):
            image_part = types.Part.from_bytes(
                data=image_bytes, mime_type=images_mime_types[i]
            )
            contents.append(f"Image {i + 1}")
            contents.append(image_part)

        if output_format is None:
            # String output
            response = self.client.models.generate_content(
                model=self.model_name.value,
                contents=contents,
            )
            return response.text if response.text else ""
        else:
            # JSON output using Google's structured output
            response = self.client.models.generate_content(
                model=self.model_name.value,
                contents=contents,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": output_format,
                },
            )
            # Fallback to text parsing if parsed is empty/null
            response_text = response.text if response.text else "{}"
            return json_loads(response_text)


# Test example
if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv

    load_dotenv()

    class ImageAnalysis(BaseModel):
        """Example custom output format for image analysis."""

        description: str
        object: str

    class AIResponse(BaseModel):
        response: list[ImageAnalysis]

    # Test with a sample image URL
    img1 = "tests/img1.png"
    img2 = "tests/img2.png"

    # Test OpenAI Vision
    try:
        print("Testing OpenAI Vision Agent...")
        openai_agent = OpenAIVisionAgent(model_name=OpenAIVisionModel.GPT_4O_MINI)
        response = openai_agent.analyze_images(
            images=[open(img1, "rb").read(), open(img2, "rb").read()],
            input_text="What do you see in these images?",
            output_format=AIResponse,
            images_mime_type=["image/png", "image/png"],
        )
        print("OpenAI Vision JSON output")
        print(response)
        print(type(response))
        assert isinstance(response, dict), "OpenAI Vision test failed"
    except Exception as e:
        print(f"OpenAI Vision test failed with exception: {e}")

    try:
        openai_agent = OpenAIVisionAgent(model_name=OpenAIVisionModel.GPT_4O_MINI)
        response = openai_agent.analyze_images(
            images=[open(img1, "rb").read(), open(img2, "rb").read()],
            input_text="What do you see in these images?",
            images_mime_type=["image/png", "image/png"],
        )
        print("OpenAI Vision Text output")
        print(response)
        print(type(response))
        assert isinstance(response, str), "OpenAI Vision test failed"
    except Exception as e:
        print(f"OpenAI Vision test failed with exception: {e}")

    # Test Anthropic Vision
    try:
        print("\nTesting Anthropic Vision Agent...")
        anthropic_agent = AnthropicVisionAgent(
            model_name=AnthropicVisionModel.CLAUDE_3_5_HAIKU
        )
        response = anthropic_agent.analyze_images(
            images=[open(img1, "rb").read(), open(img2, "rb").read()],
            input_text="What do you see in these images?",
            output_format=AIResponse,
            images_mime_type=["image/png", "image/png"],
        )
        print("Anthropic Vision JSON output")
        print(response)
        print(type(response))
        assert isinstance(response, dict), "Anthropic Vision test failed"
    except Exception as e:
        print(f"Anthropic Vision test failed with exception: {e}")

    # Test Google Vision
    try:
        print("\nTesting Google Vision Agent...")
        google_agent = GoogleVisionAgent(model_name=GoogleVisionModel.GEMINI_2_0_FLASH)
        response = google_agent.analyze_images(
            images=[open(img1, "rb").read(), open(img2, "rb").read()],
            input_text="What do you see in these images?",
            output_format=AIResponse,
            images_mime_type=["image/png", "image/png"],
        )
        print("Google Vision JSON output")
        print(response)
        print(type(response))
        assert isinstance(response, dict), "Google Vision test failed"
    except Exception as e:
        print(f"Google Vision test failed with exception: {e}")

    print("\nAll tests for VLM agents passed successfully!\n")
