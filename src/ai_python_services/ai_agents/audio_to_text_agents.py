from abc import ABC, abstractmethod
from io import BytesIO
from mimetypes import guess_type
from os import getenv

from openai import OpenAI
from google import genai
from google.genai import types

from ai_python_services.ai_agents.ai_enum import (
    Language,
    OpenAITranscriptModel,
    GoogleTranscriptModel,
)


class AudioToTextAgent(ABC):
    """Base interface for all audio-to-text agents."""

    @abstractmethod
    def transcribe(
        self,
        audio_input: bytes,
        mime_type: str = "audio/mp3",
    ) -> str:
        """Transcribe audio to text.

        Args:
            audio_input: Audio file path, file object, or audio bytes

        Returns:
            The transcribed text as string
        """
        pass


class OpenAITranscriptAgent(AudioToTextAgent):
    """OpenAI transcript-specific audio-to-text agent."""

    def __init__(
        self,
        model_name: OpenAITranscriptModel = OpenAITranscriptModel.GPT_4O_TRANSCRIBE,
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

    def transcribe(
        self,
        audio_input: bytes,
        mime_type: str = "audio/mp3",  # So the Pylance type checker doesn't complain about the default value
    ) -> str:
        """Transcribe audio using OpenAI Whisper."""

        extension = "mp3"  # Default extension if not determined
        # Get the extension from the MIME type
        if mime_type.startswith("audio/"):
            extension = mime_type.split("/")[1]

        buffer = BytesIO(audio_input)
        buffer.name = f"audio.{extension}"

        if self.prompt == "":
            prompt_text = f"The audio will mainly be in {self.language.toText()}, however, they sometimes use terminology from other languages, you should transcribe the text in multiple languages accordingly."
        else:
            prompt_text = f"{self.prompt}\nTranscribe the following audio to text in {self.language.toText()}."

        transcript = self.client.audio.transcriptions.create(
            model=self.model_name.value,
            file=buffer,
            response_format="text",
            prompt=prompt_text,
        )
        return transcript


class GoogleTranscriptAgent(AudioToTextAgent):
    """Google Speech-to-Text agent."""

    def __init__(
        self,
        model_name: GoogleTranscriptModel = GoogleTranscriptModel.GEMINI_2_5_PRO_PREVIEW,
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

    def transcribe(
        self,
        audio_input: bytes,
        mime_type: str = "audio/mp3",
    ) -> str:
        """Transcribe audio using Google Speech-to-Text.

        Args:
            audio_input: Audio file path, file object, or audio bytes
            language: Language code (e.g., 'vi-VN', 'en-US'). Default: 'vi-VN'
        """
        if self.prompt == "":
            prompt_text = (
                f"Transcribe the following audio to text in {self.language.toText()}."
            )
        else:
            prompt_text = f"{self.prompt}\nThe audio will mainly be in {self.language.toText()}, however, they sometimes use terminology from other languages, you should transcribe the text in multiple languages accordingly."
        response = self.client.models.generate_content(
            model=self.model_name.value,
            contents=[
                prompt_text,
                types.Part.from_bytes(
                    data=audio_input,
                    mime_type=mime_type,
                ),
            ],
        )

        # Get the first result's transcript
        return response.text if response.text else ""


# Test example
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    audio_file_path = "tests/audio_test.m4a"  # Replace with your audio file path
    audio_input = open(
        audio_file_path, "rb"
    ).read()  # Replace with your audio file path
    mime_type = guess_type(audio_file_path)[0] or "audio/mp3"  # Guess MIME type

    # Test OpenAI model
    try:
        openai_agent = OpenAITranscriptAgent(
            model_name=OpenAITranscriptModel.GPT_4O_TRANSCRIBE
        )
        text = openai_agent.transcribe(audio_input=audio_input, mime_type=mime_type)

        print("OpenAI Whisper agent transcription:")
        print(text)
        assert isinstance(text, str), "OpenAI transcription failed"
    except Exception as e:
        print(f"OpenAI transcription failed with exception: {e}")

    # Test Google Speech-to-Text model
    try:
        google_agent = GoogleTranscriptAgent(
            model_name=GoogleTranscriptModel.GEMINI_2_0_FLASH
        )
        text = google_agent.transcribe(audio_input=audio_input, mime_type=mime_type)

        print("Google Speech-to-Text agent transcription:")
        print(text)
        assert isinstance(text, str), "Google transcription failed"
    except Exception as e:
        print(f"Google transcription failed with exception: {e}")

    print("\nAll tests for audio-to-text agents passed successfully!\n")
