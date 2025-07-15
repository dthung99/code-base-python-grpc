"""
gRPC service implementation for AI Python Services.

This module implements the AiService gRPC service defined in ai_service.proto.
"""

import grpc
from ai_python_services.proto.ai_service import (
    ai_service_pb2,
    ai_service_pb2_grpc,
    requests_pb2,
    responses_pb2,
)


class AiServiceServicer(ai_service_pb2_grpc.AiServiceServicer):
    """Implementation of the AiService gRPC service."""

    def __init__(self):
        self.valid_api_keys = {"your-secret-api-key-here", "another-api-key"}

    def _authenticate(self, context: grpc.ServicerContext) -> bool:
        """Validate API key from metadata."""
        metadata = dict(context.invocation_metadata())
        api_key = metadata.get("api-key")

        if not api_key or api_key not in self.valid_api_keys:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid API key")
            return False
        return True

    def Health(
        self,
        request: requests_pb2.HealthRequest,
        context: grpc.ServicerContext,
    ) -> responses_pb2.HealthResponse | None:
        if not self._authenticate(context):
            return None
        # Create response message
        response = responses_pb2.HealthResponse(message="Healthy")

        return response
