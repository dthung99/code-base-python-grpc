from unittest.mock import Mock, patch

import pytest
from ai_python_services.packages.ai_agents.ai_enum import (
    AnthropicVisionModel,
    GoogleVisionModel,
    Language,
    OpenAIVisionModel,
)
from ai_python_services.packages.ai_agents.vlm_agents import (
    AnthropicVisionAgent,
    GoogleVisionAgent,
    OpenAIVisionAgent,
    VLMAgent,
)
from pydantic import BaseModel


class ImageAnalysisModel(BaseModel):
    """Test model for structured output."""

    description: str
    objects: list[str]
    confidence: float


class TestVLMAgents:
    """Test suite for Vision-Language Model agents."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.test_images = [b"fake_image_data_1", b"fake_image_data_2"]
        self.test_input_text = "Analyze these images and describe what you see"
        self.test_mime_types = ["image/png", "image/jpeg"]
        self.expected_text_response = "I can see a person and a car in the images."
        self.expected_json_response = {
            "description": "Two images showing a person and a car",
            "objects": ["person", "car"],
            "confidence": 0.95,
        }

    @pytest.fixture
    def mock_env_vars(self, monkeypatch):
        """Mock environment variables for API keys."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")

    def test_openai_vision_agent_initialization(self, mock_env_vars):
        """Test OpenAI vision agent initialization."""
        agent = OpenAIVisionAgent(
            model_name=OpenAIVisionModel.GPT_4O,
            prompt="You are an image analysis expert",
            language=Language.EN_US,
        )
        assert agent.model_name == OpenAIVisionModel.GPT_4O
        assert agent.prompt == "You are an image analysis expert"
        assert agent.language == Language.EN_US
        assert agent.client is not None

    def test_openai_vision_agent_missing_api_key(self, monkeypatch):
        """Test OpenAI vision agent raises error when API key is missing."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(
            ValueError, match="API key must be set in the environment variable 'OPENAI_API_KEY'"
        ):
            OpenAIVisionAgent()

    @patch("ai_python_services.packages.ai_agents.vlm_agents.OpenAI")
    def test_openai_vision_agent_text_analysis(self, mock_openai_class, mock_env_vars):
        """Test OpenAI vision agent text analysis."""
        # Mock the OpenAI client and response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.output_text = self.expected_text_response
        mock_client.responses.parse.return_value = mock_response

        agent = OpenAIVisionAgent(prompt="Analyze these images")
        response = agent.analyze_images(
            images=self.test_images,
            input_text=self.test_input_text,
            images_mime_type=self.test_mime_types,
        )

        assert response == self.expected_text_response
        mock_client.responses.parse.assert_called_once()

    @patch("ai_python_services.packages.ai_agents.vlm_agents.OpenAI")
    def test_openai_vision_agent_json_analysis(self, mock_openai_class, mock_env_vars):
        """Test OpenAI vision agent JSON analysis."""
        # Mock the OpenAI client and response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.output_text = (
            '{"description": "Two images showing a person and a car", '
            '"objects": ["person", "car"], "confidence": 0.95}'
        )
        mock_client.responses.parse.return_value = mock_response

        agent = OpenAIVisionAgent()
        response = agent.analyze_images(
            images=self.test_images,
            input_text=self.test_input_text,
            output_format=ImageAnalysisModel,
            images_mime_type=self.test_mime_types,
        )

        assert response == self.expected_json_response
        mock_client.responses.parse.assert_called_once()

    @patch("ai_python_services.packages.ai_agents.vlm_agents.OpenAI")
    def test_openai_vision_agent_single_mime_type(self, mock_openai_class, mock_env_vars):
        """Test OpenAI vision agent with single MIME type for multiple images."""
        # Mock the OpenAI client and response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.output_text = self.expected_text_response
        mock_client.responses.parse.return_value = mock_response

        agent = OpenAIVisionAgent()
        response = agent.analyze_images(
            images=self.test_images,
            input_text=self.test_input_text,
            images_mime_type="image/png",  # Single string instead of list
        )

        assert response == self.expected_text_response

        # Verify the content structure includes both images
        call_args = mock_client.responses.parse.call_args
        user_content = call_args[1]["input"][1]["content"]

        # Should have 4 items: 2 text items + 2 image items
        assert len(user_content) == 4
        assert user_content[0]["type"] == "input_text"
        assert user_content[1]["type"] == "input_image"
        assert user_content[2]["type"] == "input_text"
        assert user_content[3]["type"] == "input_image"

    def test_anthropic_vision_agent_initialization(self, mock_env_vars):
        """Test Anthropic vision agent initialization."""
        agent = AnthropicVisionAgent(
            model_name=AnthropicVisionModel.CLAUDE_SONNET_4,
            prompt="You are a vision expert",
            language=Language.VI_VN,
        )
        assert agent.model_name == AnthropicVisionModel.CLAUDE_SONNET_4
        assert agent.prompt == "You are a vision expert"
        assert agent.language == Language.VI_VN
        assert agent.client is not None

    def test_anthropic_vision_agent_missing_api_key(self, monkeypatch):
        """Test Anthropic vision agent raises error when API key is missing."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(
            ValueError, match="API key must be set in the environment variable 'ANTHROPIC_API_KEY'"
        ):
            AnthropicVisionAgent()

    @patch("ai_python_services.packages.ai_agents.vlm_agents.Anthropic")
    def test_anthropic_vision_agent_text_analysis(self, mock_anthropic_class, mock_env_vars):
        """Test Anthropic vision agent text analysis."""
        # Mock the Anthropic client and response
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = self.expected_text_response
        mock_client.messages.create.return_value = mock_response

        agent = AnthropicVisionAgent(prompt="Analyze these images")
        response = agent.analyze_images(
            images=self.test_images,
            input_text=self.test_input_text,
            images_mime_type=self.test_mime_types,
        )

        assert response == self.expected_text_response
        mock_client.messages.create.assert_called_once()

    @patch("ai_python_services.packages.ai_agents.vlm_agents.Anthropic")
    @patch("ai_python_services.packages.ai_agents.vlm_agents.JsonOutputParser")
    def test_anthropic_vision_agent_json_analysis(
        self, mock_parser_class, mock_anthropic_class, mock_env_vars
    ):
        """Test Anthropic vision agent JSON analysis."""
        # Mock the Anthropic client and response
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = (
            '{"description": "Two images showing a person and a car", '
            '"objects": ["person", "car"], "confidence": 0.95}'
        )
        mock_client.messages.create.return_value = mock_response

        # Mock the parser
        mock_parser = Mock()
        mock_parser.get_format_instructions.return_value = "Format as JSON"
        mock_parser.parse.return_value = self.expected_json_response
        mock_parser_class.return_value = mock_parser

        agent = AnthropicVisionAgent()
        response = agent.analyze_images(
            images=self.test_images,
            input_text=self.test_input_text,
            output_format=ImageAnalysisModel,
            images_mime_type=self.test_mime_types,
        )

        assert response == self.expected_json_response
        mock_client.messages.create.assert_called_once()
        mock_parser.parse.assert_called_once()

    @patch("ai_python_services.packages.ai_agents.vlm_agents.Anthropic")
    def test_anthropic_vision_agent_content_structure(self, mock_anthropic_class, mock_env_vars):
        """Test Anthropic vision agent creates proper content structure."""
        # Mock the Anthropic client and response
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = self.expected_text_response
        mock_client.messages.create.return_value = mock_response

        agent = AnthropicVisionAgent(prompt="Test prompt")
        agent.analyze_images(
            images=self.test_images,
            input_text=self.test_input_text,
            images_mime_type=self.test_mime_types,
        )

        # Verify the message structure
        call_args = mock_client.messages.create.call_args
        messages = call_args[1]["messages"]
        content = messages[0]["content"]

        # Should have 3 items: 1 text + 2 images
        assert len(content) == 3
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image"
        assert content[2]["type"] == "image"

        # Check image data is base64 encoded
        assert content[1]["source"]["type"] == "base64"
        assert content[1]["source"]["media_type"] == self.test_mime_types[0]
        assert content[2]["source"]["media_type"] == self.test_mime_types[1]

    def test_google_vision_agent_initialization(self, mock_env_vars):
        """Test Google vision agent initialization."""
        agent = GoogleVisionAgent(
            model_name=GoogleVisionModel.GEMINI_2_5_PRO_PREVIEW,
            prompt="You are a vision AI",
            language=Language.EN_US,
        )
        assert agent.model_name == GoogleVisionModel.GEMINI_2_5_PRO_PREVIEW
        assert agent.prompt == "You are a vision AI"
        assert agent.language == Language.EN_US
        assert agent.client is not None

    def test_google_vision_agent_missing_api_key(self, monkeypatch):
        """Test Google vision agent raises error when API key is missing."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(
            ValueError, match="API key must be set in the environment variable 'GOOGLE_API_KEY'"
        ):
            GoogleVisionAgent()

    @patch("ai_python_services.packages.ai_agents.vlm_agents.Client")
    @patch("ai_python_services.packages.ai_agents.vlm_agents.types")
    def test_google_vision_agent_text_analysis(self, mock_types, mock_genai_class, mock_env_vars):
        """Test Google vision agent text analysis."""
        # Mock the Google client and response
        mock_client = Mock()
        mock_genai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.text = self.expected_text_response
        mock_client.models.generate_content.return_value = mock_response

        # Mock the types.Part
        mock_part1 = Mock()
        mock_part2 = Mock()
        mock_types.Part.from_bytes.side_effect = [mock_part1, mock_part2]

        agent = GoogleVisionAgent(prompt="Analyze these images")
        response = agent.analyze_images(
            images=self.test_images,
            input_text=self.test_input_text,
            images_mime_type=self.test_mime_types,
        )

        assert response == self.expected_text_response
        mock_client.models.generate_content.assert_called_once()

        # Verify Part.from_bytes was called for each image
        assert mock_types.Part.from_bytes.call_count == 2

    @patch("ai_python_services.packages.ai_agents.vlm_agents.Client")
    @patch("ai_python_services.packages.ai_agents.vlm_agents.types")
    def test_google_vision_agent_json_analysis(self, mock_types, mock_genai_class, mock_env_vars):
        """Test Google vision agent JSON analysis."""
        # Mock the Google client and response
        mock_client = Mock()
        mock_genai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.text = (
            '{"description": "Two images showing a person and a car", '
            '"objects": ["person", "car"], "confidence": 0.95}'
        )
        mock_client.models.generate_content.return_value = mock_response

        # Mock the types.Part
        mock_part1 = Mock()
        mock_part2 = Mock()
        mock_types.Part.from_bytes.side_effect = [mock_part1, mock_part2]

        agent = GoogleVisionAgent()
        response = agent.analyze_images(
            images=self.test_images,
            input_text=self.test_input_text,
            output_format=ImageAnalysisModel,
            images_mime_type=self.test_mime_types,
        )

        assert response == self.expected_json_response

        # Verify structured output config
        call_args = mock_client.models.generate_content.call_args
        config = call_args[1]["config"]
        assert config["response_mime_type"] == "application/json"
        assert config["response_schema"] == ImageAnalysisModel

    @patch("ai_python_services.packages.ai_agents.vlm_agents.Client")
    @patch("ai_python_services.packages.ai_agents.vlm_agents.types")
    def test_google_vision_agent_empty_response(self, mock_types, mock_genai_class, mock_env_vars):
        """Test Google vision agent handles empty response."""
        # Mock the Google client and response
        mock_client = Mock()
        mock_genai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.text = None
        mock_client.models.generate_content.return_value = mock_response

        # Mock the types.Part
        mock_types.Part.from_bytes.return_value = Mock()

        agent = GoogleVisionAgent()
        response = agent.analyze_images(
            images=self.test_images,
            input_text=self.test_input_text,
            images_mime_type=self.test_mime_types,
        )

        assert response == ""

    @patch("ai_python_services.packages.ai_agents.vlm_agents.Client")
    @patch("ai_python_services.packages.ai_agents.vlm_agents.types")
    def test_google_vision_agent_contents_structure(
        self, mock_types, mock_genai_class, mock_env_vars
    ):
        """Test Google vision agent creates proper contents structure."""
        # Mock the Google client and response
        mock_client = Mock()
        mock_genai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.text = self.expected_text_response
        mock_client.models.generate_content.return_value = mock_response

        # Mock the types.Part
        mock_part1 = Mock()
        mock_part2 = Mock()
        mock_types.Part.from_bytes.side_effect = [mock_part1, mock_part2]

        agent = GoogleVisionAgent(prompt="Test prompt")
        agent.analyze_images(
            images=self.test_images,
            input_text=self.test_input_text,
            images_mime_type=self.test_mime_types,
        )

        # Verify the contents structure
        call_args = mock_client.models.generate_content.call_args
        contents = call_args[1]["contents"]

        # Should have 5 items: prompt + "Image 1" + part1 + "Image 2" + part2
        assert len(contents) == 5
        assert "Test prompt" in contents[0]
        assert contents[1] == "Image 1"
        assert contents[2] == mock_part1
        assert contents[3] == "Image 2"
        assert contents[4] == mock_part2

    def test_vlm_agent_interface(self):
        """Test that VLMAgent is properly defined as an abstract base class."""
        # Test that we can't instantiate the abstract base class
        with pytest.raises(TypeError):
            VLMAgent()  # type: ignore

        # Test that concrete classes implement the interface
        assert hasattr(OpenAIVisionAgent, "analyze_images")
        assert hasattr(AnthropicVisionAgent, "analyze_images")
        assert hasattr(GoogleVisionAgent, "analyze_images")

        # Test that the method signature is correct
        import inspect

        openai_sig = inspect.signature(OpenAIVisionAgent.analyze_images)
        anthropic_sig = inspect.signature(AnthropicVisionAgent.analyze_images)
        google_sig = inspect.signature(GoogleVisionAgent.analyze_images)

        # All should have the same parameter names
        assert list(openai_sig.parameters.keys()) == list(anthropic_sig.parameters.keys())
        assert list(openai_sig.parameters.keys()) == list(google_sig.parameters.keys())


if __name__ == "__main__":
    pytest.main([__file__])
