from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from ai_python_services.packages.ai_agents import (
    AnthropicAgent,
    AnthropicModel,
    GoogleAgent,
    GoogleModel,
    Language,
    OpenAIAgent,
    OpenAIModel,
)


class ResponseModel(BaseModel):
    """Response format for structured output tests."""

    name: str
    greeting: str


class TestLLMAgents:
    """Test suite for LLM agents."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.test_prompt = "You are a helpful AI assistant named Felix."
        self.test_input = "Hello, what's your name?"
        self.expected_text_response = (
            "Hello! My name is Felix. How can I help you today?"
        )
        self.expected_json_response = {
            "name": "Felix",
            "greeting": "Hello! How can I help you?",
        }

    @pytest.fixture
    def mock_env_vars(self, monkeypatch):
        """Mock environment variables for API keys."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")

    def test_openai_agent_initialization(self, mock_env_vars):
        """Test OpenAI agent initialization."""
        agent = OpenAIAgent(model_name=OpenAIModel.GPT_4O_MINI, language=Language.EN_US)
        assert agent.model_name == OpenAIModel.GPT_4O_MINI
        assert agent.language == Language.EN_US
        assert agent.client is not None

    def test_openai_agent_missing_api_key(self, monkeypatch):
        """Test OpenAI agent raises error when API key is missing."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key must be set"):
            OpenAIAgent()

    @patch("ai_python_services.packages.ai_agents.llm_agents.OpenAI")
    def test_openai_agent_text_generation(self, mock_openai_class, mock_env_vars):
        """Test OpenAI agent text generation."""
        # Mock the OpenAI client and response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = self.expected_text_response
        mock_client.chat.completions.create.return_value = mock_response

        agent = OpenAIAgent()
        response = agent.generate(self.test_prompt, self.test_input)

        assert response == self.expected_text_response
        mock_client.chat.completions.create.assert_called_once()

    @patch("ai_python_services.packages.ai_agents.llm_agents.OpenAI")
    def test_openai_agent_json_generation(self, mock_openai_class, mock_env_vars):
        """Test OpenAI agent JSON generation."""
        # Mock the OpenAI client and response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.parsed = Mock()
        mock_response.choices[
            0
        ].message.parsed.model_dump.return_value = self.expected_json_response
        mock_client.beta.chat.completions.parse.return_value = mock_response

        agent = OpenAIAgent()
        response = agent.generate(
            self.test_prompt, self.test_input, output_format=ResponseModel
        )

        assert response == self.expected_json_response
        mock_client.beta.chat.completions.parse.assert_called_once()

    def test_anthropic_agent_initialization(self, mock_env_vars):
        """Test Anthropic agent initialization."""
        agent = AnthropicAgent(
            model_name=AnthropicModel.CLAUDE_3_5_HAIKU, language=Language.VI_VN
        )
        assert agent.model_name == AnthropicModel.CLAUDE_3_5_HAIKU
        assert agent.language == Language.VI_VN
        assert agent.client is not None

    def test_anthropic_agent_missing_api_key(self, monkeypatch):
        """Test Anthropic agent raises error when API key is missing."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key must be set"):
            AnthropicAgent()

    @patch("ai_python_services.packages.ai_agents.llm_agents.Anthropic")
    def test_anthropic_agent_text_generation(self, mock_anthropic_class, mock_env_vars):
        """Test Anthropic agent text generation."""
        # Mock the Anthropic client and response
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = self.expected_text_response
        mock_client.messages.create.return_value = mock_response

        agent = AnthropicAgent()
        response = agent.generate(self.test_prompt, self.test_input)

        assert response == self.expected_text_response
        mock_client.messages.create.assert_called_once()

    def test_google_agent_initialization(self, mock_env_vars):
        """Test Google agent initialization."""
        agent = GoogleAgent(
            model_name=GoogleModel.GEMINI_2_0_FLASH, language=Language.EN_US
        )
        assert agent.model_name == GoogleModel.GEMINI_2_0_FLASH
        assert agent.language == Language.EN_US
        assert agent.client is not None

    def test_google_agent_missing_api_key(self, monkeypatch):
        """Test Google agent raises error when API key is missing."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key must be set"):
            GoogleAgent()

    @patch("ai_python_services.packages.ai_agents.llm_agents.genai.Client")
    def test_google_agent_text_generation(self, mock_genai_class, mock_env_vars):
        """Test Google agent text generation."""
        # Mock the Google client and response
        mock_client = Mock()
        mock_genai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.text = self.expected_text_response
        mock_client.models.generate_content.return_value = mock_response

        agent = GoogleAgent()
        response = agent.generate(self.test_prompt, self.test_input)

        assert response == self.expected_text_response
        mock_client.models.generate_content.assert_called_once()

    @patch("ai_python_services.packages.ai_agents.llm_agents.genai.Client")
    def test_google_agent_json_generation(self, mock_genai_class, mock_env_vars):
        """Test Google agent JSON generation."""
        # Mock the Google client and response
        mock_client = Mock()
        mock_genai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.text = (
            '{"name": "Felix", "greeting": "Hello! How can I help you?"}'
        )
        mock_client.models.generate_content.return_value = mock_response

        agent = GoogleAgent()
        response = agent.generate(
            self.test_prompt, self.test_input, output_format=ResponseModel
        )

        assert response == self.expected_json_response
        mock_client.models.generate_content.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
