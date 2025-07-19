import os
from unittest.mock import Mock, patch

import grpc
from ai_python_services.core.constants.grpc_metadata import GrpcMetadata
from ai_python_services.core.decorators.auth import RequireAuth
from grpc import StatusCode


class TestRequireAuth:
    """Test cases for the RequireAuth decorator class."""

    def test_init_with_custom_api_keys(self):
        """Test initialization with custom API keys."""
        custom_keys = {"key1", "key2", "key3"}
        auth_decorator = RequireAuth(api_keys=custom_keys)

        assert auth_decorator.api_keys == custom_keys

    @patch.dict(
        os.environ, {"GRPC_SECRET_API_KEY_1": "test_key_1", "GRPC_SECRET_API_KEY_2": "test_key_2"}
    )
    def test_init_with_environment_variables(self):
        """Test initialization with environment variables."""
        auth_decorator = RequireAuth()

        expected_keys = {"test_key_1", "test_key_2"}
        assert auth_decorator.api_keys == expected_keys

    @patch.dict(os.environ, {}, clear=True)
    def test_init_with_empty_environment(self):
        """Test initialization with empty environment variables."""
        auth_decorator = RequireAuth()

        # Should contain empty strings when env vars are not set
        expected_keys = {""}
        assert auth_decorator.api_keys == expected_keys

    def test_decorator_with_valid_api_key(self):
        """Test decorator allows request with valid API key."""
        # Setup
        valid_keys = {"valid_key_1", "valid_key_2"}
        auth_decorator = RequireAuth(api_keys=valid_keys)

        # Mock function to decorate
        mock_func = Mock(return_value="success")
        decorated_func = auth_decorator(mock_func)

        # Mock gRPC context
        mock_context = Mock(spec=grpc.ServicerContext)
        mock_context.invocation_metadata.return_value = [(GrpcMetadata.API_KEY, "valid_key_1")]

        # Mock service instance and request
        mock_service = Mock()
        mock_request = Mock()

        # Call decorated function
        result = decorated_func(mock_service, mock_request, mock_context)

        # Assertions
        assert result == "success"
        mock_func.assert_called_once_with(mock_service, mock_request, mock_context)
        mock_context.abort.assert_not_called()

    def test_decorator_with_invalid_api_key(self):
        """Test decorator rejects request with invalid API key."""
        # Setup
        valid_keys = {"valid_key_1", "valid_key_2"}
        auth_decorator = RequireAuth(api_keys=valid_keys)

        # Mock function to decorate
        mock_func = Mock()
        decorated_func = auth_decorator(mock_func)

        # Mock gRPC context
        mock_context = Mock(spec=grpc.ServicerContext)
        mock_context.invocation_metadata.return_value = [(GrpcMetadata.API_KEY, "invalid_key")]

        # Mock service instance and request
        mock_service = Mock()
        mock_request = Mock()

        # Call decorated function
        result = decorated_func(mock_service, mock_request, mock_context)

        # Assertions
        assert result is None
        mock_func.assert_not_called()
        mock_context.abort.assert_called_once_with(StatusCode.UNAUTHENTICATED, "Invalid API key")

    def test_decorator_with_missing_api_key(self):
        """Test decorator rejects request with missing API key."""
        # Setup
        valid_keys = {"valid_key_1", "valid_key_2"}
        auth_decorator = RequireAuth(api_keys=valid_keys)

        # Mock function to decorate
        mock_func = Mock()
        decorated_func = auth_decorator(mock_func)

        # Mock gRPC context with no API key metadata
        mock_context = Mock(spec=grpc.ServicerContext)
        mock_context.invocation_metadata.return_value = [("other-metadata", "some_value")]

        # Mock service instance and request
        mock_service = Mock()
        mock_request = Mock()

        # Call decorated function
        result = decorated_func(mock_service, mock_request, mock_context)

        # Assertions
        assert result is None
        mock_func.assert_not_called()
        mock_context.abort.assert_called_once_with(StatusCode.UNAUTHENTICATED, "Invalid API key")

    def test_decorator_with_empty_api_key(self):
        """Test decorator rejects request with empty API key."""
        # Setup
        valid_keys = {"valid_key_1", "valid_key_2"}
        auth_decorator = RequireAuth(api_keys=valid_keys)

        # Mock function to decorate
        mock_func = Mock()
        decorated_func = auth_decorator(mock_func)

        # Mock gRPC context with empty API key
        mock_context = Mock(spec=grpc.ServicerContext)
        mock_context.invocation_metadata.return_value = [(GrpcMetadata.API_KEY, "")]

        # Mock service instance and request
        mock_service = Mock()
        mock_request = Mock()

        # Call decorated function
        result = decorated_func(mock_service, mock_request, mock_context)

        # Assertions
        assert result is None
        mock_func.assert_not_called()
        mock_context.abort.assert_called_once_with(StatusCode.UNAUTHENTICATED, "Invalid API key")

    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves original function metadata."""
        # Setup
        auth_decorator = RequireAuth(api_keys={"test_key"})

        def sample_function(self, request, context):
            """Sample function docstring."""
            return "sample_result"

        # Decorate function
        decorated_func = auth_decorator(sample_function)

        # Assertions
        assert decorated_func.__name__ == sample_function.__name__
        assert decorated_func.__doc__ == sample_function.__doc__


class TestRequireAuthInstance:
    """Test cases for the require_auth singleton instance."""

    def test_require_auth_singleton(self):
        """Test that require_auth is properly initialized."""
        # Test that require_auth is an instance of RequireAuth
        require_auth = RequireAuth()
        assert isinstance(require_auth, RequireAuth)

        # Test that it uses environment variables
        expected_keys = {"test-grpc-secret-api-key-1", "test-grpc-secret-api-key-2"}
        assert require_auth.api_keys == expected_keys

    def test_require_auth_decorator_usage(self):
        """Test using require_auth as a decorator."""
        # Mock function to decorate
        require_auth = RequireAuth()
        mock_func = Mock(return_value="decorated_result")

        # Use require_auth as decorator
        decorated_func = require_auth(mock_func)

        # Mock gRPC context with valid key (assuming env vars are set)
        mock_context = Mock(spec=grpc.ServicerContext)
        mock_context.invocation_metadata.return_value = [
            (GrpcMetadata.API_KEY, next(iter(require_auth.api_keys)))  # Use first valid key
        ]

        # Mock service instance and request
        mock_service = Mock()
        mock_request = Mock()

        # Call decorated function
        result = decorated_func(mock_service, mock_request, mock_context)

        # Assertions
        assert result == "decorated_result"
        mock_func.assert_called_once_with(mock_service, mock_request, mock_context)


class TestAuthIntegration:
    """Integration tests for auth decorator with realistic scenarios."""

    def test_multiple_metadata_entries(self):
        """Test decorator works correctly with multiple metadata entries."""
        # Setup
        auth_decorator = RequireAuth(api_keys={"correct_key"})
        mock_func = Mock(return_value="success")
        decorated_func = auth_decorator(mock_func)

        # Mock gRPC context with multiple metadata entries
        mock_context = Mock(spec=grpc.ServicerContext)
        mock_context.invocation_metadata.return_value = [
            ("user-agent", "grpc-client"),
            ("content-type", "application/grpc"),
            (GrpcMetadata.API_KEY, "correct_key"),
            ("x-custom-header", "custom-value"),
        ]

        # Mock service and request
        mock_service = Mock()
        mock_request = Mock()

        # Call decorated function
        result = decorated_func(mock_service, mock_request, mock_context)

        # Assertions
        assert result == "success"
        mock_func.assert_called_once()
        mock_context.abort.assert_not_called()

    def test_case_sensitive_api_key_metadata(self):
        """Test that API key metadata lookup is case-sensitive."""
        # Setup
        auth_decorator = RequireAuth(api_keys={"valid_key"})
        mock_func = Mock()
        decorated_func = auth_decorator(mock_func)

        # Mock gRPC context with wrong case metadata key
        mock_context = Mock(spec=grpc.ServicerContext)
        mock_context.invocation_metadata.return_value = [
            ("API-KEY", "valid_key"),  # Wrong case
            ("Api-Key", "valid_key"),  # Wrong case
        ]

        # Mock service and request
        mock_service = Mock()
        mock_request = Mock()

        # Call decorated function
        result = decorated_func(mock_service, mock_request, mock_context)

        # Should fail because metadata key case doesn't match
        assert result is None
        mock_func.assert_not_called()
        mock_context.abort.assert_called_once_with(StatusCode.UNAUTHENTICATED, "Invalid API key")
