from enum import Enum


class Language(Enum):
    """Enum for supported languages."""

    VI_VN = "vi-VN"
    EN_US = "en-US"

    def toText(self) -> str:
        """Convert enum value to a more human-readable string."""
        # Define a mapping for the display names
        display_names = {
            Language.VI_VN: "Vietnamese",
            Language.EN_US: "English",
            # Add other mappings if you extend the enum
        }
        return display_names.get(self, self.value)


class OpenAIModel(Enum):
    """Enum for OpenAI model names."""

    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
    GPT_4_1 = "gpt-4.1"


class AnthropicModel(Enum):
    """Enum for Anthropic model names."""

    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-20241022"
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet-20250219"
    CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"
    CLAUDE_OPUS_4 = "claude-opus-4-20250514"


class GoogleModel(Enum):
    """Enum for Google model names."""

    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_2_5_FLASH_PREVIEW = "gemini-2.5-flash-preview-05-20"
    GEMINI_2_5_PRO_PREVIEW = "gemini-2.5-pro-preview-06-05"


class OpenAIVisionModel(Enum):
    """Enum for OpenAI vision model names."""

    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
    GPT_4_1 = "gpt-4.1"


class AnthropicVisionModel(Enum):
    """Enum for Anthropic vision model names."""

    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-20241022"
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet-20250219"
    CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"
    CLAUDE_OPUS_4 = "claude-opus-4-20250514"


class GoogleVisionModel(Enum):
    """Enum for Google vision model names."""

    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_2_5_FLASH_PREVIEW = "gemini-2.5-flash-preview-05-20"
    GEMINI_2_5_PRO_PREVIEW = "gemini-2.5-pro-preview-06-05"


class OpenAITranscriptModel(Enum):
    """Enum for OpenAI transcript model names."""

    GPT_4O_TRANSCRIBE = "gpt-4o-transcribe"


class GoogleTranscriptModel(Enum):
    """Enum for Google Speech-to-Text model names."""

    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_2_5_FLASH_PREVIEW = "gemini-2.5-flash-preview-05-20"
    GEMINI_2_5_PRO_PREVIEW = "gemini-2.5-pro-preview-06-05"
