"""
gRPC service implementation for AI Python Services.

This module implements the AiService gRPC service defined in ai_service.proto.
"""

from concurrent import futures

import grpc
from grpc_reflection.v1alpha import reflection  # Add this import

from ai_python_services.proto import ai_service_pb2, ai_service_pb2_grpc


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

    def SayHello(
        self,
        request: ai_service_pb2.HelloRequest,
        context: grpc.ServicerContext,
    ) -> ai_service_pb2.HelloResponse | None:
        """
        Simple hello world RPC method.

        Args:
            request: HelloRequest containing the name
            context: gRPC context for the request

        Returns:
            HelloResponse with greeting message
        """

        if not self._authenticate(context):
            return

        print(f"📥 Received SayHello request: name='{request.name}'")

        # Create response message
        message = f"Hello {request.name}! This is the AI Python gRPC Service 🤖"
        response = ai_service_pb2.HelloResponse(message=message)

        print(f"📤 Sending response: message='{response.message}'")
        return response

    def SayHelloNext(
        self,
        request: ai_service_pb2.HelloRequest,
        context: grpc.ServicerContext,
    ) -> ai_service_pb2.HelloResponse | None:
        """
        Simple hello world RPC method.

        Args:
            request: HelloRequest containing the name
            context: gRPC context for the request

        Returns:
            HelloResponse with greeting message
        """

        if not self._authenticate(context):
            return

        print(f"📥 Received SayHello request: name='{request.name}'")

        # Create response message
        message = f"Hello {request.name}! This is the AI Python gRPC Service 🤖"
        response = ai_service_pb2.HelloResponse(message=message)

        print(f"📤 Sending response: message='{response.message}'")
        return response
