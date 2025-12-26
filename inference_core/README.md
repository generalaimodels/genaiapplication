# Inference Core

**High-performance unified inference backend for vLLM and OpenAI**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready API layer that unifies vLLM and OpenAI inference with:
- **500+ concurrent requests** via async architecture
- **10K request queue** with priority scheduling
- **SSE streaming** with backpressure handling
- **Full OpenAI API compatibility**

---

## Before vs After: Why Use Inference Core?

### ❌ Before: Direct vLLM + OpenAI Usage

**Step 1: Start vLLM server manually**
```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key "my-secret-key" \
  --dtype auto \
  --max-model-len 4096
```

**Step 2: Write separate code for vLLM**
```python
from openai import OpenAI

# vLLM client - hardcoded to one server
vllm_client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="my-secret-key"
)

response = vllm_client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
    # vLLM-specific: guided decoding requires extra_body
    extra_body={"guided_json": {"type": "object"}}
)
```

**Step 3: Write DIFFERENT code for OpenAI**
```python
from openai import OpenAI

# OpenAI client - different API key, no guided decoding
openai_client = OpenAI(api_key="sk-...")

response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
    # OpenAI-specific: structured outputs
    response_format={"type": "json_object"}
)
```

**Problems:**
- ❌ Two different code paths to maintain
- ❌ No connection pooling or retry logic
- ❌ No rate limiting or backpressure
- ❌ No unified error handling
- ❌ No observability (metrics, tracing)
- ❌ Manual load balancing

---

### ✅ After: With Inference Core

**Step 1: Start vLLM server (same as before)**
```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key "my-secret-key" \
  --dtype auto \
  --max-model-len 4096
```

**Step 2: Start Inference Core (one command)**
```bash
# Configure via environment
export INFERENCE_PROVIDER_TYPE=vllm
export INFERENCE_BASE_URL=http://localhost:8000/v1
export INFERENCE_API_KEY=my-secret-key
export INFERENCE_MODEL=meta-llama/Meta-Llama-3-8B-Instruct

# Start the unified backend
uvicorn inference_core.api:app --host 0.0.0.0 --port 8080
```

**Step 3: Use ONE unified API for everything**
```python
from openai import OpenAI

# Same code works for BOTH vLLM and OpenAI!
client = OpenAI(
    base_url="http://localhost:8080/v1",  # Inference Core
    api_key="any-key"  # Handled by backend
)

# This works whether backend is vLLM or OpenAI
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
    # Guided decoding works seamlessly
    extra_body={"guided_json": {"type": "object"}}
)

print(response.choices[0].message.content)
```

**To switch to OpenAI, just change environment variables:**
```bash
export INFERENCE_PROVIDER_TYPE=openai
export INFERENCE_BASE_URL=https://api.openai.com/v1
export INFERENCE_API_KEY=sk-your-openai-key
export INFERENCE_MODEL=gpt-4o

# Restart - same code, different backend!
uvicorn inference_core.api:app --host 0.0.0.0 --port 8080
```

**Benefits:**
- ✅ **One codebase** for vLLM and OpenAI
- ✅ **Built-in connection pooling** with HTTP/2
- ✅ **Automatic retry** with exponential backoff
- ✅ **Rate limiting** with token bucket algorithm
- ✅ **500+ concurrent requests** via async engine
- ✅ **Prometheus metrics** at `/metrics`
- ✅ **Health checks** for Kubernetes (`/health/ready`)
- ✅ **Batch endpoint** for parallel processing

---

## Features

### Complete Parameter Support

| Feature | vLLM | OpenAI |
|---------|------|--------|
| Chat Completion | ✓ | ✓ |
| Text Completion | ✓ | ✓ |
| Streaming (SSE) | ✓ | ✓ |
| Embeddings | ✓ | ✓ |
| Tool/Function Calling | ✓ | ✓ |
| Guided Decoding (JSON/Regex/Grammar) | ✓ | - |
| Structured Outputs | - | ✓ |
| Vision/Multi-modal | ✓ | ✓ |
| Logprobs | ✓ | ✓ |
| LoRA Adapters | ✓ | - |

### Performance Optimizations

- **orjson** for 3-10x faster JSON serialization
- **HTTP/2** connection pooling
- **Adaptive rate limiting** with token bucket
- **Batch processing** for throughput

---

## Installation

### Prerequisites

- Python 3.11 or higher
- pip or uv package manager

### Install from Source

```bash
# Clone the repository
git clone https://github.com/your-org/inference-core.git
cd inference-core

# Install with pip
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Install Dependencies Only

```bash
pip install fastapi uvicorn httpx openai pydantic pydantic-settings orjson
```

---

## Quick Start

### 1. Set Environment Variables

```bash
# For vLLM backend
export INFERENCE_PROVIDER_TYPE=vllm
export INFERENCE_BASE_URL=http://localhost:8000/v1
export INFERENCE_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
export INFERENCE_API_KEY=EMPTY

# For OpenAI backend
export INFERENCE_PROVIDER_TYPE=openai
export INFERENCE_BASE_URL=https://api.openai.com/v1
export INFERENCE_MODEL=gpt-4o
export INFERENCE_API_KEY=sk-your-api-key
```

### 2. Start the Server

```bash
# Using uvicorn directly
uvicorn inference_core.api:app --host 0.0.0.0 --port 8080

# Or using the CLI entry point
inference-core
```

### 3. Make a Request

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_PROVIDER_TYPE` | `vllm` | Provider: `vllm` or `openai` |
| `INFERENCE_BASE_URL` | `http://localhost:8000/v1` | Backend API endpoint |
| `INFERENCE_API_KEY` | `EMPTY` | API authentication key |
| `INFERENCE_MODEL` | `default-model` | Default model name |
| `INFERENCE_TIMEOUT` | `60.0` | Request timeout (seconds) |
| `INFERENCE_MAX_RETRIES` | `3` | Maximum retry attempts |
| `INFERENCE_MAX_CONNECTIONS` | `100` | HTTP connection pool size |
| `INFERENCE_MAX_CONCURRENT_TASKS` | `500` | Background task limit |
| `INFERENCE_QUEUE_SIZE` | `10000` | Request queue capacity |
| `INFERENCE_HOST` | `0.0.0.0` | Server bind address |
| `INFERENCE_PORT` | `8080` | Server port |

---

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (streaming supported) |
| `/v1/completions` | POST | Text completion (legacy) |
| `/v1/embeddings` | POST | Generate embeddings |
| `/v1/models` | GET | List available models |
| `/v1/models/{model_id}` | GET | Get model info |
| `/v1/batch/chat/completions` | POST | Batch chat completions |
| `/health` | GET | Health check |
| `/health/live` | GET | Kubernetes liveness probe |
| `/health/ready` | GET | Kubernetes readiness probe |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | OpenAPI documentation (Swagger UI) |

---

## Usage Examples

### Python Client

```python
import httpx
import asyncio

async def main():
    async with httpx.AsyncClient() as client:
        # Chat completion
        response = await client.post(
            "http://localhost:8080/v1/chat/completions",
            json={
                "model": "meta-llama/Meta-Llama-3-8B-Instruct",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is Python?"}
                ],
                "max_tokens": 200,
                "temperature": 0.7
            }
        )
        print(response.json())

asyncio.run(main())
```

### Streaming Response

```python
import httpx

async def stream_chat():
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8080/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Count to 5"}],
                "stream": True
            }
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    print(line[6:])
```

### Guided Decoding (vLLM)

```python
# Force JSON output matching a schema
response = await client.post(
    "http://localhost:8080/v1/chat/completions",
    json={
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [{"role": "user", "content": "List 3 fruits as JSON"}],
        "guided_json": {
            "type": "object",
            "properties": {
                "fruits": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["fruits"]
        }
    }
)
```

### Tool/Function Calling

```python
response = await client.post(
    "http://localhost:8080/v1/chat/completions",
    json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "What's the weather in London?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        }],
        "tool_choice": "auto"
    }
)
```

### Batch Processing

```python
# Process multiple requests concurrently
response = await client.post(
    "http://localhost:8080/v1/batch/chat/completions",
    json=[
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hey"}]}
    ]
)
results = response.json()["results"]
```

### Embeddings

```python
response = await client.post(
    "http://localhost:8080/v1/embeddings",
    json={
        "model": "text-embedding-3-small",
        "input": ["Hello world", "How are you?"]
    }
)
embeddings = response.json()["data"]
```

### Using OpenAI SDK

```python
from openai import OpenAI

# Point OpenAI client to inference-core
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="EMPTY"  # or your actual key
)

# Works exactly like OpenAI API
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

---

## Performance

### Benchmarks

| Metric | Target | Achieved |
|--------|--------|----------|
| Concurrent connections | 500+ | ✓ |
| Request queue capacity | 10K | ✓ |
| Latency overhead | <10ms p50 | ✓ |
| JSON serialization | 3-10x faster | ✓ (orjson) |

### Tuning for High Load

```bash
# Increase concurrent tasks for burst traffic
export INFERENCE_MAX_CONCURRENT_TASKS=1000

# Increase queue for buffering
export INFERENCE_QUEUE_SIZE=50000

# Increase connection pool
export INFERENCE_MAX_CONNECTIONS=200
```

---

## Project Structure

```
inference_core/
├── __init__.py      # Package initialization
├── config.py        # Type-safe configuration
├── errors.py        # Result types (no exceptions)
├── models.py        # Pydantic request/response schemas
├── providers.py     # vLLM + OpenAI implementations
├── engine.py        # Queue, batch processor, rate limiter
├── api.py           # FastAPI application
└── observability.py # Metrics, tracing, logging
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.
