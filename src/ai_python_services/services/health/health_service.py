"""
gRPC service implementation for AI Python Services.

This module implements the AiService gRPC service defined in ai_service.proto.
"""

import grpc

from ai_python_services.core.decorators.auth import require_auth
from ai_python_services.proto.health import (
    health_pb2_grpc,
    health_requests_pb2,
    health_responses_pb2,
)


class HealthServiceServicer(health_pb2_grpc.HealthServiceServicer):
    """Implementation of the HealthService gRPC service."""

    def __init__(self):
        pass

    def Health(
        self,
        request: health_requests_pb2.HealthRequest,
        context: grpc.ServicerContext,
    ) -> health_responses_pb2.HealthResponse:
        # Create response message
        response = health_responses_pb2.HealthResponse(message="Healthy")

        return response

    @require_auth
    def HealthWithAuthentication(
        self,
        request: health_requests_pb2.HealthRequest,
        context: grpc.ServicerContext,
    ) -> health_responses_pb2.HealthResponse:
        # Create response message
        response = health_responses_pb2.HealthResponse(message="Healthy")

        return response
