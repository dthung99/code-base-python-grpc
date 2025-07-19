# src/ai_python_services/core/decorators/auth.py
import os
from functools import wraps
from typing import Callable, Optional, Set

import grpc

from ai_python_services.core.constants.grpc_metadata import GrpcMetadata


class RequireAuth:
    """Class-based authentication decorator."""

    def __init__(self, api_keys: Optional[Set[str]] = None):
        """
        Initialize the decorator with API keys.

        Args:
            api_keys: Optional set of valid API keys. If None, reads from environment.
        """
        print("Initializing authentication class")
        if api_keys is None:
            self.api_keys = {
                os.getenv("GRPC_SECRET_API_KEY_1", ""),
                os.getenv("GRPC_SECRET_API_KEY_2", ""),
            }
        else:
            self.api_keys = api_keys

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self_service, request, context: grpc.ServicerContext):
            # Extract metadata
            metadata = dict(context.invocation_metadata())
            api_key = metadata.get(GrpcMetadata.API_KEY)

            # Validate API key
            if not api_key or api_key not in self.api_keys:
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid API key")
                return None

            # Call the original method
            return func(self_service, request, context)

        return wrapper


require_auth = RequireAuth()
