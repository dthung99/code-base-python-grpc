from unittest.mock import Mock, patch

import grpc
import pytest
from ai_python_services.core.constants.grpc_metadata import GrpcMetadata
from ai_python_services.proto.health import (
    health_requests_pb2,
    health_responses_pb2,
)
from ai_python_services.services.health.health_service import HealthServiceServicer
from grpc import StatusCode


class TestHealthServiceServicer:
    """Test cases for the HealthServiceServicer."""

    @pytest.fixture
    def health_service(self):
        """Create a HealthServiceServicer instance for testing."""
        return HealthServiceServicer()

    @pytest.fixture
    def mock_context(self):
        """Create a mock gRPC context."""
        return Mock(spec=grpc.ServicerContext)

    @pytest.fixture
    def health_request(self):
        """Create a HealthRequest for testing."""
        return health_requests_pb2.HealthRequest()

    def test_health_service_initialization(self, health_service):
        """Test that HealthServiceServicer initializes correctly."""
        assert isinstance(health_service, HealthServiceServicer)

    def test_health_endpoint_success(self, health_service, health_request, mock_context):
        """Test the Health endpoint returns successful response."""
        # Call the Health method
        response = health_service.Health(health_request, mock_context)

        # Assertions
        assert isinstance(response, health_responses_pb2.HealthResponse)
        assert response.message == "Healthy"

        # Verify context wasn't aborted
        mock_context.abort.assert_not_called()

    def test_health_endpoint_with_empty_request(self, health_service, mock_context):
        """Test the Health endpoint with empty request."""
        # Create empty request
        empty_request = health_requests_pb2.HealthRequest()

        # Call the Health method
        response = health_service.Health(empty_request, mock_context)

        # Assertions
        assert isinstance(response, health_responses_pb2.HealthResponse)
        assert response.message == "Healthy"

    def test_health_with_authentication_valid_key(
        self, health_service, health_request, mock_context
    ):
        """Test HealthWithAuthentication endpoint with valid authentication."""

        # Setup context with valid API key
        mock_context.invocation_metadata.return_value = [
            (GrpcMetadata.API_KEY, "test-grpc-secret-api-key-1")
        ]

        # Call the HealthWithAuthentication method
        response = health_service.HealthWithAuthentication(health_request, mock_context)

        # Assertions
        assert isinstance(response, health_responses_pb2.HealthResponse)
        assert response.message == "Healthy"
        mock_context.abort.assert_not_called()

    def test_health_with_authentication_integration(self, health_request, mock_context):
        """Test HealthWithAuthentication with actual auth decorator integration."""
        # Create health service instance
        health_service = HealthServiceServicer()

        # Setup context with valid API key (temporarily add to require_auth)
        test_key = "integration_test_key"

        # Import the actual require_auth singleton
        from ai_python_services.core.decorators.auth import require_auth

        # Temporarily add test key
        original_keys = require_auth.api_keys.copy()
        require_auth.api_keys.add(test_key)

        try:
            # Setup context with valid key
            mock_context.invocation_metadata.return_value = [(GrpcMetadata.API_KEY, test_key)]

            # Call the method
            response = health_service.HealthWithAuthentication(health_request, mock_context)

            # Assertions
            assert isinstance(response, health_responses_pb2.HealthResponse)
            assert response.message == "Healthy"
            mock_context.abort.assert_not_called()

        finally:
            # Restore original keys
            require_auth.api_keys = original_keys

    def test_health_with_authentication_invalid_key(self, health_request, mock_context):
        """Test HealthWithAuthentication with invalid authentication."""
        # Create health service instance
        health_service = HealthServiceServicer()

        # Setup context with invalid API key
        mock_context.invocation_metadata.return_value = [
            (GrpcMetadata.API_KEY, "definitely_invalid_key_12345")
        ]

        # Call the method
        response = health_service.HealthWithAuthentication(health_request, mock_context)

        # Assertions - should be None due to auth failure
        assert response is None
        mock_context.abort.assert_called_once_with(StatusCode.UNAUTHENTICATED, "Invalid API key")

    def test_health_with_authentication_missing_key(self, health_request, mock_context):
        """Test HealthWithAuthentication with missing API key."""
        # Create health service instance
        health_service = HealthServiceServicer()

        # Setup context with no API key
        mock_context.invocation_metadata.return_value = [("other-header", "some-value")]

        # Call the method
        response = health_service.HealthWithAuthentication(health_request, mock_context)

        # Assertions - should be None due to auth failure
        assert response is None
        mock_context.abort.assert_called_once_with(StatusCode.UNAUTHENTICATED, "Invalid API key")

    def test_health_response_message_content(self, health_service, health_request, mock_context):
        """Test that health response contains expected message content."""
        response = health_service.Health(health_request, mock_context)

        # Test message content specifically
        assert response.message == "Healthy"
        assert len(response.message) > 0
        assert isinstance(response.message, str)

    def test_health_service_methods_exist(self, health_service):
        """Test that required service methods exist and are callable."""
        # Test that methods exist
        assert hasattr(health_service, "Health")
        assert hasattr(health_service, "HealthWithAuthentication")

        # Test that methods are callable
        assert callable(health_service.Health)
        assert callable(health_service.HealthWithAuthentication)

    def test_multiple_health_calls(self, health_service, health_request, mock_context):
        """Test multiple calls to health endpoint return consistent results."""
        # Call health multiple times
        response1 = health_service.Health(health_request, mock_context)
        response2 = health_service.Health(health_request, mock_context)
        response3 = health_service.Health(health_request, mock_context)

        # All responses should be identical
        assert response1.message == response2.message == response3.message
        assert all(
            isinstance(r, health_responses_pb2.HealthResponse)
            for r in [response1, response2, response3]
        )

    def test_health_service_stateless(self):
        """Test that health service is stateless - multiple instances behave the same."""
        service1 = HealthServiceServicer()
        service2 = HealthServiceServicer()

        request = health_requests_pb2.HealthRequest()
        context = Mock(spec=grpc.ServicerContext)

        response1 = service1.Health(request, context)
        response2 = service2.Health(request, context)

        # Both instances should return the same result
        assert response1.message == response2.message


class TestHealthServiceIntegration:
    """Integration tests for health service with realistic scenarios."""

    def test_health_service_with_grpc_metadata(self):
        """Test health service with realistic gRPC metadata."""
        health_service = HealthServiceServicer()
        request = health_requests_pb2.HealthRequest()

        # Mock context with realistic metadata
        mock_context = Mock(spec=grpc.ServicerContext)
        mock_context.invocation_metadata.return_value = [
            ("user-agent", "grpc-python/1.50.0"),
            ("content-type", "application/grpc"),
            ("grpc-accept-encoding", "gzip"),
        ]

        # Should work without API key for regular health check
        response = health_service.Health(request, mock_context)

        assert response.message == "Healthy"
        mock_context.abort.assert_not_called()

    def test_authenticated_health_with_realistic_metadata(self):
        """Test authenticated health endpoint with realistic metadata."""
        health_service = HealthServiceServicer()
        request = health_requests_pb2.HealthRequest()

        # Import and setup test key
        from ai_python_services.core.decorators.auth import require_auth

        test_key = "realistic_api_key_abc123"
        original_keys = require_auth.api_keys.copy()
        require_auth.api_keys.add(test_key)

        try:
            # Mock context with realistic metadata including API key
            mock_context = Mock(spec=grpc.ServicerContext)
            mock_context.invocation_metadata.return_value = [
                ("user-agent", "grpc-python/1.50.0"),
                ("content-type", "application/grpc"),
                (GrpcMetadata.API_KEY, test_key),
                ("grpc-accept-encoding", "gzip"),
                ("x-request-id", "req-12345"),
            ]

            response = health_service.HealthWithAuthentication(request, mock_context)

            assert response.message == "Healthy"
            mock_context.abort.assert_not_called()

        finally:
            require_auth.api_keys = original_keys
