from unittest.mock import Mock

import grpc
import pytest
from ai_python_services.proto.ai_service import (
    ai_service_pb2,
    ai_service_pb2_grpc,
    ai_service_requests_pb2,
    ai_service_responses_pb2,
)
from ai_python_services.services.ai_service import AiServiceServicer


class TestAiServiceServicer:
    """Test suite for AiService gRPC servicer."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.servicer = AiServiceServicer()
        self.valid_api_key = "your-secret-api-key-here"
        self.invalid_api_key = "invalid-key"

    def create_mock_context(self, api_key: str | None = None) -> Mock:
        """Create a mock gRPC context with metadata."""
        context = Mock(spec=grpc.ServicerContext)

        if api_key:
            # Mock the invocation_metadata to return API key
            context.invocation_metadata.return_value = [("api-key", api_key)]
        else:
            # No API key provided
            context.invocation_metadata.return_value = []

        return context

    def test_servicer_initialization(self):
        """Test that servicer initializes with valid API keys."""
        assert self.servicer.valid_api_keys == {"your-secret-api-key-here", "another-api-key"}
        assert len(self.servicer.valid_api_keys) == 2

    def test_authenticate_with_valid_api_key(self):
        """Test authentication succeeds with valid API key."""
        context = self.create_mock_context(self.valid_api_key)

        result = self.servicer._authenticate(context)

        assert result is True
        context.abort.assert_not_called()

    def test_authenticate_with_invalid_api_key(self):
        """Test authentication fails with invalid API key."""
        context = self.create_mock_context(self.invalid_api_key)

        result = self.servicer._authenticate(context)

        assert result is False
        context.abort.assert_called_once_with(grpc.StatusCode.UNAUTHENTICATED, "Invalid API key")

    def test_authenticate_with_no_api_key(self):
        """Test authentication fails when no API key is provided."""
        context = self.create_mock_context()  # No API key

        result = self.servicer._authenticate(context)

        assert result is False
        context.abort.assert_called_once_with(grpc.StatusCode.UNAUTHENTICATED, "Invalid API key")

    def test_authenticate_with_none_api_key(self):
        """Test authentication fails when API key is None."""
        context = Mock(spec=grpc.ServicerContext)
        context.invocation_metadata.return_value = [("api-key", None)]

        result = self.servicer._authenticate(context)

        assert result is False
        context.abort.assert_called_once_with(grpc.StatusCode.UNAUTHENTICATED, "Invalid API key")

    def test_health_with_valid_authentication(self):
        """Test Health method returns success with valid authentication."""
        request = ai_service_requests_pb2.HealthRequest()
        context = self.create_mock_context(self.valid_api_key)

        response = self.servicer.Health(request, context)

        assert response is not None
        assert isinstance(response, ai_service_responses_pb2.HealthResponse)
        assert response.message == "Healthy"
        context.abort.assert_not_called()

    def test_health_with_invalid_authentication(self):
        """Test Health method returns None when authentication fails."""
        request = ai_service_requests_pb2.HealthRequest()
        context = self.create_mock_context(self.invalid_api_key)

        response = self.servicer.Health(request, context)

        assert response is None
        context.abort.assert_called_once_with(grpc.StatusCode.UNAUTHENTICATED, "Invalid API key")

    def test_health_request_object(self):
        """Test that HealthRequest can be created properly."""
        request = ai_service_requests_pb2.HealthRequest()
        assert request is not None
        assert isinstance(request, ai_service_requests_pb2.HealthRequest)

    def test_health_response_object(self):
        """Test that HealthResponse can be created and has correct message."""
        response = ai_service_responses_pb2.HealthResponse(message="Test message")
        assert response is not None
        assert isinstance(response, ai_service_responses_pb2.HealthResponse)
        assert response.message == "Test message"

    def test_metadata_parsing(self):
        """Test that metadata is parsed correctly from context."""
        # Test with multiple metadata items
        context = Mock(spec=grpc.ServicerContext)
        context.invocation_metadata.return_value = [
            ("api-key", self.valid_api_key),
            ("user-agent", "test-client"),
            ("content-type", "application/grpc"),
        ]

        result = self.servicer._authenticate(context)

        assert result is True
        # Verify invocation_metadata was called to get the metadata
        context.invocation_metadata.assert_called_once()

    def test_case_sensitive_api_key_header(self):
        """Test that API key header is case sensitive."""
        context = Mock(spec=grpc.ServicerContext)
        # Use uppercase header name
        context.invocation_metadata.return_value = [("API-KEY", self.valid_api_key)]

        result = self.servicer._authenticate(context)

        # Should fail because we're looking for lowercase "api-key"
        assert result is False
        context.abort.assert_called_once_with(grpc.StatusCode.UNAUTHENTICATED, "Invalid API key")

    def test_multiple_api_keys_in_metadata(self):
        """Test behavior when multiple api-key headers are present."""
        context = Mock(spec=grpc.ServicerContext)
        # Multiple api-key headers - dict() will take the last one
        context.invocation_metadata.return_value = [
            ("api-key", self.invalid_api_key),
            ("api-key", self.valid_api_key),
        ]

        result = self.servicer._authenticate(context)

        # Should succeed because dict() takes the last value
        assert result is True

    def test_valid_api_keys_set(self):
        """Test that valid API keys are stored as a set for O(1) lookup."""
        assert isinstance(self.servicer.valid_api_keys, set)
        assert "your-secret-api-key-here" in self.servicer.valid_api_keys
        assert "another-api-key" in self.servicer.valid_api_keys
        assert "invalid-key" not in self.servicer.valid_api_keys

    def test_servicer_inherits_from_generated_base(self):
        """Test that servicer inherits from the generated gRPC base class."""
        assert isinstance(self.servicer, ai_service_pb2_grpc.AiServiceServicer)

    def test_health_method_signature(self):
        """Test that Health method has correct signature."""
        import inspect

        sig = inspect.signature(self.servicer.Health)
        params = list(sig.parameters.keys())

        # Should have self, request, and context parameters
        assert params == ["request", "context"]

        # Check return type annotation
        return_annotation = sig.return_annotation
        expected_return = ai_service_responses_pb2.HealthResponse | None
        assert return_annotation == expected_return

    # @patch("ai_python_services.services.ai_service.grpc.StatusCode.UNAUTHENTICATED")
    def test_authentication_uses_correct_status_code(self):
        """Test that authentication failure uses correct gRPC status code."""
        # mock_status_code.UNAUTHENTICATED = grpc.StatusCode.UNAUTHENTICATED
        context = self.create_mock_context(self.invalid_api_key)

        self.servicer._authenticate(context)

        context.abort.assert_called_once()
        args, kwargs = context.abort.call_args
        assert args[0] == grpc.StatusCode.UNAUTHENTICATED
        assert args[1] == "Invalid API key"

    def test_context_abort_not_called_on_success(self):
        """Test that context.abort is not called when authentication succeeds."""
        request = ai_service_requests_pb2.HealthRequest()
        context = self.create_mock_context(self.valid_api_key)

        response = self.servicer.Health(request, context)

        assert response is not None
        assert response.message == "Healthy"
        context.abort.assert_not_called()

    def test_empty_metadata_dict(self):
        """Test authentication with empty metadata dictionary."""
        context = Mock(spec=grpc.ServicerContext)
        context.invocation_metadata.return_value = []

        result = self.servicer._authenticate(context)

        assert result is False
        context.abort.assert_called_once_with(grpc.StatusCode.UNAUTHENTICATED, "Invalid API key")


class TestProtobufMessages:
    """Test protobuf message generation and serialization."""

    def test_health_request_serialization(self):
        """Test HealthRequest can be serialized and deserialized."""
        original_request = ai_service_requests_pb2.HealthRequest()

        # Serialize to bytes
        serialized = original_request.SerializeToString()
        assert isinstance(serialized, bytes)

        # Deserialize back
        deserialized_request = ai_service_requests_pb2.HealthRequest()
        deserialized_request.ParseFromString(serialized)

        # Should be equivalent (empty messages)
        assert original_request == deserialized_request

    def test_health_response_serialization(self):
        """Test HealthResponse can be serialized and deserialized."""
        original_response = ai_service_responses_pb2.HealthResponse(message="Test Health")

        # Serialize to bytes
        serialized = original_response.SerializeToString()
        assert isinstance(serialized, bytes)

        # Deserialize back
        deserialized_response = ai_service_responses_pb2.HealthResponse()
        deserialized_response.ParseFromString(serialized)

        # Should have same message
        assert deserialized_response.message == "Test Health"
        assert original_response == deserialized_response

    def test_health_response_default_values(self):
        """Test HealthResponse default values."""
        response = ai_service_responses_pb2.HealthResponse()

        # Message should be empty string by default
        assert response.message == ""

        # Test setting message
        response.message = "Custom message"
        assert response.message == "Custom message"

    def test_protobuf_field_access(self):
        """Test that protobuf fields can be accessed and modified."""
        response = ai_service_responses_pb2.HealthResponse()

        # Test field exists
        assert hasattr(response, "message")

        # Test field can be set
        response.message = "Field test"
        assert response.message == "Field test"

        # Test field can be cleared
        response.ClearField("message")
        assert response.message == ""


if __name__ == "__main__":
    pytest.main([__file__])
