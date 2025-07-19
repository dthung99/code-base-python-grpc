# tests/unit/conftest.py
from os import getenv

import pytest
from ai_python_services.core.decorators.auth import require_auth
from dotenv import load_dotenv


@pytest.fixture(scope="session", autouse=True)
def load_test_env():
    """Load test environment variables."""
    print("Loading test environment variables...")
    load_dotenv(".env.test", override=True)  # Loads from .env in project root
    # Set up RequireAuth keys from environment variables
    require_auth.api_keys = {
        getenv("GRPC_SECRET_API_KEY_1", ""),
        getenv("GRPC_SECRET_API_KEY_2", ""),
    }
