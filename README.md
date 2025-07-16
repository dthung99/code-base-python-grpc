# Python gRPC Services Codebase

A Python gRPC service codebase with AI services as example implementation. Built with modern Python tooling for type-safe, scalable backend services.

## Quick Start

```bash
git clone <repository-url>
cd ai-python
uv sync
uv run python scripts/generate_proto.py
uv run python main.py
```

Server runs on `localhost:50051` with API key authentication.

## Structure

```
├── proto/                    # Protocol buffer definitions
├── src/ai_python_services/
│   ├── packages/            # Reusable packages (AI agents example)
│   ├── proto/               # Generated gRPC code
│   └── services/            # gRPC service implementations
├── scripts/generate_proto.py # Proto code generation
└── tests/                   # Test suite
```

## Core Features

- **gRPC Framework**: Authentication, error handling, type safety
- **Proto Management**: Automated code generation with type stubs
- **Example AI Services**: Text generation, image analysis, audio transcription
- **Modern Tooling**: uv package manager, pytest, full type hints

## Example AI Services

Supports OpenAI, Anthropic, and Google models for:

- Text generation (LLM)
- Image analysis (VLM)
- Audio transcription

Set API keys in `.env`:

```env
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
GOOGLE_API_KEY=your_key
```

## Client Usage (Node.js)

```javascript
const client = new aiService.AiService(
  "localhost:50051",
  grpc.credentials.createInsecure()
);
const metadata = new grpc.Metadata();
metadata.set("api-key", "your-secret-api-key-here");

client.Health({}, metadata, (error, response) => {
  console.log("Health:", response.message);
});
```

## Development

```bash
# Tests
uv run pytest

# Regenerate proto code
uv run python scripts/generate_proto.py
```
