"""AI Agents Package - A collection of AI model wrappers for different providers."""

from .ai_enum import (
    Language,
    OpenAIModel,
    AnthropicModel,
    GoogleModel,
    OpenAIVisionModel,
    AnthropicVisionModel,
    GoogleVisionModel,
    OpenAITranscriptModel,
    GoogleTranscriptModel,
)

from .llm_agents import (
    LLMAgent,
    OpenAIAgent,
    AnthropicAgent,
    GoogleAgent,
)

from .vlm_agents import (
    VLMAgent,
    OpenAIVisionAgent,
    AnthropicVisionAgent,
    GoogleVisionAgent,
)

from .audio_to_text_agents import (
    AudioToTextAgent,
    OpenAITranscriptAgent,
    GoogleTranscriptAgent,
)

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "dthung.y17@gmail.com"

__all__ = [
    # Enums
    "Language",
    "OpenAIModel",
    "AnthropicModel",
    "GoogleModel",
    "OpenAIVisionModel",
    "AnthropicVisionModel",
    "GoogleVisionModel",
    "OpenAITranscriptModel",
    "GoogleTranscriptModel",
    # LLM Agents
    "LLMAgent",
    "OpenAIAgent",
    "AnthropicAgent",
    "GoogleAgent",
    # VLM Agents
    "VLMAgent",
    "OpenAIVisionAgent",
    "AnthropicVisionAgent",
    "GoogleVisionAgent",
    # Audio-to-Text Agents
    "AudioToTextAgent",
    "OpenAITranscriptAgent",
    "GoogleTranscriptAgent",
]
