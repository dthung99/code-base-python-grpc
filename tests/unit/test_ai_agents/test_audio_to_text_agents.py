from unittest.mock import Mock, patch

import pytest
from ai_python_services.packages.ai_agents.ai_enum import (
    GoogleTranscriptModel,
    Language,
    OpenAITranscriptModel,
)
from ai_python_services.packages.ai_agents.audio_to_text_agents import (
    AudioToTextAgent,
    GoogleTranscriptAgent,
    OpenAITranscriptAgent,
)


class TestAudioToTextAgents:
    """Test suite for audio-to-text agents."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.test_audio_bytes = b"fake_audio_data"
        self.test_mime_type = "audio/mp3"
        self.expected_transcript = "This is a test transcription"

    @pytest.fixture
    def mock_env_vars(self, monkeypatch):
        """Mock environment variables for API keys."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")

    def test_openai_transcript_agent_initialization(self, mock_env_vars):
        """Test OpenAI transcript agent initialization."""
        agent = OpenAITranscriptAgent(
            model_name=OpenAITranscriptModel.GPT_4O_TRANSCRIBE,
            prompt="Test prompt",
            language=Language.EN_US,
        )
        assert agent.model_name == OpenAITranscriptModel.GPT_4O_TRANSCRIBE
        assert agent.prompt == "Test prompt"
        assert agent.language == Language.EN_US
        assert agent.client is not None

    def test_openai_transcript_agent_missing_api_key(self, monkeypatch):
        """Test OpenAI transcript agent raises error when API key is missing."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(
            ValueError, match="API key must be set in the environment variable 'OPENAI_API_KEY'"
        ):
            OpenAITranscriptAgent()

    @patch("ai_python_services.packages.ai_agents.audio_to_text_agents.OpenAI")
    def test_openai_transcript_agent_transcription_default_prompt(
        self, mock_openai_class, mock_env_vars
    ):
        """Test OpenAI transcript agent transcription with default prompt."""
        # Mock the OpenAI client and response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.audio.transcriptions.create.return_value = self.expected_transcript

        agent = OpenAITranscriptAgent(prompt="", language=Language.VI_VN)
        response = agent.transcribe(self.test_audio_bytes, self.test_mime_type)

        assert response == self.expected_transcript
        mock_client.audio.transcriptions.create.assert_called_once()

        # Verify the call arguments
        call_args = mock_client.audio.transcriptions.create.call_args
        assert call_args[1]["model"] == OpenAITranscriptModel.GPT_4O_TRANSCRIBE.value
        assert call_args[1]["response_format"] == "text"
        assert "Vietnamese" in call_args[1]["prompt"]

    @patch("ai_python_services.packages.ai_agents.audio_to_text_agents.OpenAI")
    def test_openai_transcript_agent_transcription_custom_prompt(
        self, mock_openai_class, mock_env_vars
    ):
        """Test OpenAI transcript agent transcription with custom prompt."""
        # Mock the OpenAI client and response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.audio.transcriptions.create.return_value = self.expected_transcript

        custom_prompt = "Please transcribe this medical audio"
        agent = OpenAITranscriptAgent(prompt=custom_prompt, language=Language.EN_US)
        response = agent.transcribe(self.test_audio_bytes, self.test_mime_type)

        assert response == self.expected_transcript

        # Verify the call arguments contain custom prompt
        call_args = mock_client.audio.transcriptions.create.call_args
        assert custom_prompt in call_args[1]["prompt"]
        assert "English" in call_args[1]["prompt"]

    @patch("ai_python_services.packages.ai_agents.audio_to_text_agents.OpenAI")
    def test_openai_transcript_agent_different_mime_types(self, mock_openai_class, mock_env_vars):
        """Test OpenAI transcript agent with different MIME types."""
        # Mock the OpenAI client and response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.audio.transcriptions.create.return_value = self.expected_transcript

        agent = OpenAITranscriptAgent()

        # Test with different MIME types
        test_cases = [
            ("audio/mp3", "mp3"),
            ("audio/wav", "wav"),
            ("audio/m4a", "m4a"),
            ("audio/flac", "flac"),
        ]

        for mime_type, expected_extension in test_cases:
            response = agent.transcribe(self.test_audio_bytes, mime_type)
            assert response == self.expected_transcript

            # Verify the file buffer name has correct extension
            call_args = mock_client.audio.transcriptions.create.call_args
            file_buffer = call_args[1]["file"]
            assert file_buffer.name == f"audio.{expected_extension}"

    def test_google_transcript_agent_initialization(self, mock_env_vars):
        """Test Google transcript agent initialization."""
        agent = GoogleTranscriptAgent(
            model_name=GoogleTranscriptModel.GEMINI_2_0_FLASH,
            prompt="Test prompt",
            language=Language.VI_VN,
        )
        assert agent.model_name == GoogleTranscriptModel.GEMINI_2_0_FLASH
        assert agent.prompt == "Test prompt"
        assert agent.language == Language.VI_VN
        assert agent.client is not None

    def test_google_transcript_agent_missing_api_key(self, monkeypatch):
        """Test Google transcript agent raises error when API key is missing."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(
            ValueError, match="API key must be set in the environment variable 'GOOGLE_API_KEY'"
        ):
            GoogleTranscriptAgent()

    @patch("ai_python_services.packages.ai_agents.audio_to_text_agents.Client")
    def test_google_transcript_agent_transcription_default_prompt(
        self, mock_genai_class, mock_env_vars
    ):
        """Test Google transcript agent transcription with default prompt."""
        # Mock the Google client and response
        mock_client = Mock()
        mock_genai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.text = self.expected_transcript
        mock_client.models.generate_content.return_value = mock_response

        agent = GoogleTranscriptAgent(prompt="", language=Language.EN_US)
        response = agent.transcribe(self.test_audio_bytes, self.test_mime_type)

        assert response == self.expected_transcript
        mock_client.models.generate_content.assert_called_once()

        # Verify the call arguments
        call_args = mock_client.models.generate_content.call_args
        assert call_args[1]["model"] == GoogleTranscriptModel.GEMINI_2_5_PRO_PREVIEW.value
        contents = call_args[1]["contents"]
        assert "English" in contents[0]

    @patch("ai_python_services.packages.ai_agents.audio_to_text_agents.Client")
    def test_google_transcript_agent_transcription_custom_prompt(
        self, mock_genai_class, mock_env_vars
    ):
        """Test Google transcript agent transcription with custom prompt."""
        # Mock the Google client and response
        mock_client = Mock()
        mock_genai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.text = self.expected_transcript
        mock_client.models.generate_content.return_value = mock_response

        custom_prompt = "Please transcribe this Vietnamese audio"
        agent = GoogleTranscriptAgent(prompt=custom_prompt, language=Language.VI_VN)
        response = agent.transcribe(self.test_audio_bytes, self.test_mime_type)

        assert response == self.expected_transcript

        # Verify the call arguments contain custom prompt
        call_args = mock_client.models.generate_content.call_args
        contents = call_args[1]["contents"]
        assert custom_prompt in contents[0]
        assert "Vietnamese" in contents[0]

    @patch("ai_python_services.packages.ai_agents.audio_to_text_agents.Client")
    def test_google_transcript_agent_empty_response(self, mock_genai_class, mock_env_vars):
        """Test Google transcript agent handles empty response."""
        # Mock the Google client and response
        mock_client = Mock()
        mock_genai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.text = None
        mock_client.models.generate_content.return_value = mock_response

        agent = GoogleTranscriptAgent()
        response = agent.transcribe(self.test_audio_bytes, self.test_mime_type)

        assert response == ""

    @patch("ai_python_services.packages.ai_agents.audio_to_text_agents.types")
    @patch("ai_python_services.packages.ai_agents.audio_to_text_agents.Client")
    def test_google_transcript_agent_part_creation(
        self, mock_genai_class, mock_types, mock_env_vars
    ):
        """Test Google transcript agent creates proper Part object."""
        # Mock the Google client and response
        mock_client = Mock()
        mock_genai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.text = self.expected_transcript
        mock_client.models.generate_content.return_value = mock_response

        mock_part = Mock()
        mock_types.Part.from_bytes.return_value = mock_part

        agent = GoogleTranscriptAgent()
        response = agent.transcribe(self.test_audio_bytes, self.test_mime_type)

        assert response == self.expected_transcript

        # Verify Part.from_bytes was called with correct parameters
        mock_types.Part.from_bytes.assert_called_once_with(
            data=self.test_audio_bytes, mime_type=self.test_mime_type
        )

    def test_audio_to_text_agent_interface(self):
        """Test that AudioToTextAgent is properly defined as an abstract base class."""
        # Test that we can't instantiate the abstract base class
        with pytest.raises(TypeError):
            AudioToTextAgent()  # type: ignore

        # Test that concrete classes implement the interface
        assert hasattr(OpenAITranscriptAgent, "transcribe")
        assert hasattr(GoogleTranscriptAgent, "transcribe")

        # Test that the method signature is correct
        import inspect

        openai_sig = inspect.signature(OpenAITranscriptAgent.transcribe)
        google_sig = inspect.signature(GoogleTranscriptAgent.transcribe)

        # Both should have the same parameter names
        assert list(openai_sig.parameters.keys()) == list(google_sig.parameters.keys())


if __name__ == "__main__":
    pytest.main([__file__])
