# vLLM Server & OpenAI Client Integration Guide

A comprehensive guide for serving Large Language Models (LLMs) with **vLLM** and interacting with them using the **OpenAI** Python client for various text-to-text, vision, and advanced generation tasks.

---

## Table of Contents

- [Overview](#overview)
- [Core Concept](#core-concept)
- [Prerequisites](#prerequisites)
- [Server Side Setup](#server-side-setup)
  - [Basic Command](#basic-command)
  - [Production Command](#production-command)
  - [Key Arguments Reference](#key-arguments-reference)
- [Client Side Setup](#client-side-setup)
  - [Installation](#installation)
  - [Global Client Configuration](#global-client-configuration)
- [Text Generation Tasks](#text-generation-tasks)
  - [Task 1: Chat Completion](#task-1-chat-completion-chatbot-style)
  - [Task 2: Text Completion](#task-2-text-completion-raw-generation)
  - [Task 3: Streaming Responses](#task-3-streaming-responses-real-time)
  - [Task 4: Structured Output (JSON Mode)](#task-4-structured-output-json-mode)
  - [Task 5: Batch Processing](#task-5-batch-processing-multiple-prompts)
- [Advanced Tasks](#advanced-tasks)
  - [Task 6: Text Embeddings](#task-6-text-embeddings)
  - [Task 7: Function Calling (Tool Use)](#task-7-function-calling-tool-use)
  - [Task 8: Multi-LoRA Serving](#task-8-multi-lora-serving)
  - [Task 9: Guided Decoding (Regex/Choice)](#task-9-guided-decoding-regexchoice)
- [Multi-Modal Tasks](#multi-modal-tasks-vision-language)
  - [Vision Model Server Setup](#1-server-side-serve-vision-model)
  - [Image via URL](#option-a-image-via-url-easiest)
  - [Image via Base64](#option-b-image-via-base64-local-images)
  - [Important Notes for Multi-Modal](#important-notes-for-multi-modal)
- [API Reference Summary](#api-reference-summary)
- [Quick Start Example](#quick-start-example)

---

## Overview

This documentation details how to:
- **Serve models** using vLLM as a high-performance inference server
- **Interact with models** using the standard OpenAI Python client
- **Perform various tasks** including chat, completion, streaming, JSON output, batch processing, embeddings, function calling, LoRA adapters, guided decoding, and vision-language tasks

---

## Core Concept

```
┌─────────────────────────────────────────────────────────────────┐
│                        ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────────┐         ┌──────────────────────────┐     │
│   │   CLIENT SIDE    │  HTTP   │      SERVER SIDE         │     │
│   │                  │ ──────► │                          │     │
│   │  OpenAI Python   │         │   vLLM Inference Server  │     │
│   │     Library      │ ◄────── │   (GPU-accelerated)      │     │
│   │                  │         │                          │     │
│   └──────────────────┘         └──────────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

| Component | Description |
|-----------|-------------|
| **Server Side (`vllm`)** | Loads the model into GPU memory and exposes it via an HTTP server. Acts as a **drop-in replacement** for the OpenAI API. |
| **Client Side (`openai`)** | Uses the standard OpenAI Python library to send requests to your local vLLM server instead of OpenAI's servers. |

---

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU with sufficient VRAM
- vLLM installed (`pip install vllm`)
- OpenAI Python library (`pip install openai`)

---

## Server Side Setup

Start the vLLM server before making any client requests.

### Basic Command

```bash
vllm serve <model_name>
```

**Example:**
```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct
```

### Production Command

Recommended configuration with performance and compatibility flags:

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key "my-secret-key" \
  --dtype auto \
  --max-model-len 4096
```

### Key Arguments Reference

| Argument | Description | Default |
|----------|-------------|---------|
| `--host` | Host IP address | `localhost` |
| | `0.0.0.0` → External access | |
| | `localhost` → Local only | |
| `--port` | Port number | `8000` |
| `--api-key` | Custom API key for authentication | `"EMPTY"` |
| `--gpu-memory-utilization` | Fraction of GPU memory to use | `0.90` |
| `--max-model-len` | Context window size (reduce for OOM errors) | Model default |
| `--dtype` | Data type for model weights | `auto` |
| `--trust-remote-code` | Required for many custom/vision models | `False` |
| `--enable-lora` | Enable LoRA adapter support | `False` |
| `--enable-auto-tool-choice` | Enable automatic tool/function calling | `False` |

---

## Client Side Setup

### Installation

```bash
pip install openai
```

### Global Client Configuration

Use this setup block for **all** tasks below:

```python
from openai import OpenAI

# Initialize client pointing to your local vLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",  # URL where vllm serve is running
    api_key="my-secret-key",              # Must match --api-key flag (or "EMPTY")
)

# The model name in the client MUST match the model name used in 'vllm serve'
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
```

> ⚠️ **Important:** The `model_name` must exactly match the model loaded on the server.

---

## Text Generation Tasks

### Task 1: Chat Completion (Chatbot Style)

**Use Case:** Conversational agents, customer support bots, Q&A systems

```python
response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain quantum computing in one sentence."}
    ],
    temperature=0.7,
    max_tokens=100
)

print(response.choices[0].message.content)
```

**Output Example:**
```
Quantum computing uses quantum bits (qubits) that can exist in multiple states 
simultaneously, enabling parallel processing of complex calculations exponentially 
faster than classical computers.
```

---

### Task 2: Text Completion (Raw Generation)

**Use Case:** Code completion, story continuation, non-conversational generation

> 📌 **Note:** This targets the `/v1/completions` endpoint.

```python
response = client.completions.create(
    model=model_name,
    prompt="def fibonacci(n):",
    max_tokens=64,
    temperature=0.1  # Lower temperature is better for code
)

print(response.choices[0].text)
```

**Output Example:**
```python
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    else:
        fib = [0, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return fib
```

---

### Task 3: Streaming Responses (Real-time)

**Use Case:** UI applications requiring low latency, ChatGPT-like typing effect

```python
stream = client.chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": "Write a short poem about rust."}],
    stream=True,  # Enable streaming
)

print("Generating: ", end="", flush=True)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # Newline at end
```

**Output Behavior:**
```
Generating: In metal's heart, a quiet fight,
Orange blooms where moisture bites.
Time's patient hand, relentless, slow,
Transforms the strong to dust below.
```
*(Text appears token by token in real-time)*

---

### Task 4: Structured Output (JSON Mode)

**Use Case:** Data extraction, API integration, machine-readable output

> 📌 **Note:** The prompt must explicitly ask for JSON for best results.

```python
response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "You are a data extractor. Output JSON only."},
        {"role": "user", "content": "Extract data: John Doe is 30 years old."}
    ],
    # Enforce JSON format (vLLM supports this via guided decoding)
    extra_body=dict(guided_json={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
    })
)

print(response.choices[0].message.content)
```

**Output Example:**
```json
{
  "name": "John Doe",
  "age": 30
}
```

---

### Task 5: Batch Processing (Multiple Prompts)

**Use Case:** Processing multiple unrelated prompts for higher throughput

```python
# Simple loop approach for batch processing
prompts = [
    "What is the capital of France?",
    "What is 2+2?",
    "Who wrote Hamlet?"
]

for prompt in prompts:
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    print(f"Q: {prompt}")
    print(f"A: {response.choices[0].message.content}\n")
```

**Output Example:**
```
Q: What is the capital of France?
A: The capital of France is Paris.

Q: What is 2+2?
A: 2+2 equals 4.

Q: Who wrote Hamlet?
A: Hamlet was written by William Shakespeare.
```

---

## Advanced Tasks

### Task 6: Text Embeddings

**Use Case:** RAG (Retrieval Augmented Generation), semantic search, similarity matching

> ⚠️ **Important:** You must serve an embedding-specific model (e.g., `BAAI/bge-m3` or `e5-mistral`), not a chat model.

#### Server Command

```bash
# You must use a dedicated embedding model
vllm serve BAAI/bge-m3 --task embed
```

#### Client Code

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

# Request embeddings for a list of texts
response = client.embeddings.create(
    model="BAAI/bge-m3",
    input=["Hello world", "Machine learning is fascinating"]
)

# Print the vector for the first input
print(f"Vector length: {len(response.data[0].embedding)}")
print(f"First 5 dimensions: {response.data[0].embedding[:5]}")
```

**Output Example:**
```
Vector length: 1024
First 5 dimensions: [0.0234, -0.0156, 0.0891, -0.0423, 0.0567]
```

---

### Task 7: Function Calling (Tool Use)

**Use Case:** Building agents that can "act" (e.g., get weather, query databases, execute commands)

> 📌 **Note:** vLLM supports the standard OpenAI tool usage format if the model (like Llama 3 or Mistral) supports it.

#### Server Command

```bash
# Ensure you use a model trained for tool use (e.g., Mistral-Instruct, Llama-3-Instruct)
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json
```

> 📌 **Note:** `--tool-call-parser` is optional if vLLM detects the correct template automatically, but `llama3_json` or `mistral` is safer for specific models.

#### Client Code

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City, e.g. London"}
            },
            "required": ["location"]
        }
    }
}]

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "What is the weather in Paris?"}],
    tools=tools
)

# The model will return a "tool_call" instead of plain text
tool_call = response.choices[0].message.tool_calls[0]
print(f"Function to call: {tool_call.function.name}")
print(f"Arguments: {tool_call.function.arguments}")
```

**Output Example:**
```
Function to call: get_weather
Arguments: {"location": "Paris"}
```

---

### Task 8: Multi-LoRA Serving

**Use Case:** Serve one base model with multiple specialized adapters, switch dynamically per request

vLLM allows you to serve **one base model** and **multiple adapters** (LoRAs) simultaneously. You can call different adapters dynamically in your client code.

#### Server Command

You map names (aliases) to the file paths of your adapters:

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --enable-lora \
  --lora-modules sql_adapter=/path/to/sql-lora-v1 \
                 poet_adapter=/path/to/poetry-lora-v1
```

#### Client Code

To use a specific adapter, simply pass its **alias** as the `model` parameter:

```python
# Request 1: Use the SQL Adapter
response_sql = client.chat.completions.create(
    model="sql_adapter",  # Matches the alias in --lora-modules
    messages=[{"role": "user", "content": "Select all users."}]
)

# Request 2: Use the Poet Adapter
response_poet = client.chat.completions.create(
    model="poet_adapter",  # Matches the other alias
    messages=[{"role": "user", "content": "Write a sonnet about clouds."}]
)

print("SQL Output:", response_sql.choices[0].message.content)
print("Poem Output:", response_poet.choices[0].message.content)
```

**Output Example:**
```
SQL Output: SELECT * FROM users;
Poem Output: Upon the azure canvas, soft and wide,
The clouds drift by like dreams upon the breeze...
```

---

### Task 9: Guided Decoding (Regex/Choice)

**Use Case:** Strict formatting requirements that standard prompting (or even JSON mode) fails to satisfy

You can force the model to follow a **Regex** pattern or choose from a specific list.

#### Server Command

Standard serve command (no special flags needed):

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct
```

#### Client Code (Regex Enforcement)

Forces the output to match a pattern (e.g., an IP address, email, or specific date format):

```python
# Force the model to output a valid IP address only
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Generate a fake server IP address."}],
    extra_body={
        "guided_regex": r"\b(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    }
)

print(response.choices[0].message.content)
```

**Output Example:**
```
192.168.1.105
```

#### Client Code (Choice Enforcement)

Force the model to pick *only* from a predefined list:

```python
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Is this email spam or not? Email: 'Win $1000 now!'"}],
    extra_body={
        "guided_choice": ["SPAM", "NOT_SPAM"]
    }
)

# The output is guaranteed to be exactly one of the strings above
print(response.choices[0].message.content)
```

**Output Example:**
```
SPAM
```

---

## Multi-Modal Tasks (Vision-Language)

This section covers **Visual Question Answering (VQA)** or **Image Description**, where you send both an image and text to the model.

> 📌 **Note:** vLLM support for vision models (like LLaVA, Phi-3.5-Vision, Qwen-VL) is highly active. Ensure you are on the latest version of vLLM.

### 1. Server Side: Serve Vision Model

You must serve a specific vision-language model. **LLaVA-1.5** is a reliable standard for this.

```bash
# Serve the LLaVA-1.5 model
# --trust-remote-code: Required for many vision models
# --limit-mm-per-prompt: Limits images per request (usually 1 for basic models)
vllm serve llava-hf/llava-1.5-7b-hf \
  --trust-remote-code \
  --limit-mm-per-prompt image=1
```

**Alternative for better performance:**
```bash
vllm serve microsoft/Phi-3.5-vision-instruct \
  --trust-remote-code \
  --limit-mm-per-prompt image=1
```

### 2. Client Side: Vision Code Snippets

To send an image, you must change the `content` field from a simple string to a **list of dictionaries**. You can pass images via a public **URL** or a **Base64** string.

#### Option A: Image via URL (Easiest)

Use this if your image is hosted online:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

# Model name must match the one you served
model_name = "llava-hf/llava-1.5-7b-hf"

response = client.chat.completions.create(
    model=model_name,
    messages=[{
        "role": "user",
        "content": [
            # 1. The Text Prompt
            {"type": "text", "text": "What is in this image? Describe it in detail."},
            # 2. The Image URL
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                },
            },
        ],
    }],
    max_tokens=300
)

print(response.choices[0].message.content)
```

**Output Example:**
```
The image shows a wooden boardwalk extending through a lush green meadow or marsh. 
The sky above is blue with scattered white clouds. The grass on either side of the 
boardwalk is tall and green, suggesting this is during spring or summer. The 
boardwalk appears to be a nature trail, likely in a park or nature preserve.
```

#### Option B: Image via Base64 (Local Images)

Use this if you want to upload a local file from your computer:

```python
import base64
from openai import OpenAI

# Function to encode local image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

# Load and encode your local image
image_path = "./my_local_image.jpg"  # Replace with your file path
base64_image = encode_image(image_path)

response = client.chat.completions.create(
    model="llava-hf/llava-1.5-7b-hf",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What color is the object in this picture?"},
            {
                "type": "image_url",
                "image_url": {
                    # Format: data:image/jpeg;base64,{base64_string}
                    "url": f"data:image/jpeg;base64,{base64_image}"
                },
            },
        ],
    }],
    max_tokens=100
)

print(response.choices[0].message.content)
```

### Important Notes for Multi-Modal

| Consideration | Details |
|---------------|---------|
| **Prompt Format** | Unlike standard text models, you do **not** need to manually insert `<image>` tokens in your text string when using the OpenAI client. The API server handles the merging of the image and text for you. |
| **Single vs. Multi-Image** | Most vLLM vision implementations currently support **one image per prompt** reliably. If you need multi-image support (e.g., comparing two photos), use specific models like `Phi-3.5-vision` and set `--limit-mm-per-prompt image=2` (or higher). |
| **Memory** | Vision models use significantly more GPU memory (VRAM) because they must load a visual encoder (like CLIP) alongside the language model. If you get OOM errors, try reducing `--gpu-memory-utilization 0.8`. |

---

## API Reference Summary

| Task | Endpoint | OpenAI Method | Use Case |
|------|----------|---------------|----------|
| **Chat** | `/v1/chat/completions` | `client.chat.completions.create` | Chatbots, Assistants |
| **Completion** | `/v1/completions` | `client.completions.create` | Code autocompletion, Novel writing |
| **JSON** | `/v1/chat/completions` | `create` + `extra_body` | Data extraction, API integration |
| **Streaming** | `/v1/chat/completions` | `create(stream=True)` | UI applications requiring low latency |
| **Embeddings** | `/v1/embeddings` | `client.embeddings.create` | RAG, Semantic search |
| **Tool Use** | `/v1/chat/completions` | `create` + `tools` | Agents, Function calling |
| **LoRA** | `/v1/chat/completions` | `create(model="adapter_alias")` | Multi-adapter serving |
| **Guided Regex** | `/v1/chat/completions` | `create` + `extra_body` | Strict format enforcement |
| **Guided Choice** | `/v1/chat/completions` | `create` + `extra_body` | Classification, Constrained outputs |
| **Vision** | `/v1/chat/completions` | `create` + image content | VQA, Image description |

---

## Quick Start Example

Complete end-to-end example:

### Terminal 1 - Start Server

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key "my-secret-key"
```

### Terminal 2 - Run Client

```python
from openai import OpenAI

# Setup
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="my-secret-key",
)
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# Make request
response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! What can you do?"}
    ],
    temperature=0.7,
    max_tokens=150
)

print(response.choices[0].message.content)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **OOM (Out of Memory)** | Reduce `--max-model-len` or `--gpu-memory-utilization 0.8` |
| **Connection Refused** | Ensure server is running and check host/port settings |
| **Model Name Mismatch** | Client `model` parameter must exactly match served model |
| **API Key Error** | Ensure client `api_key` matches server `--api-key` flag |
| **Tool Calling Not Working** | Add `--enable-auto-tool-choice` and appropriate `--tool-call-parser` |

---

## License

This documentation is provided as-is for educational purposes.

---

**Happy Inferencing! 🚀**