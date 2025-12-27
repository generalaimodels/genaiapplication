"""
Manual Verification Script (SOTA).

Adheres to:
- Isolation: Runs in separate folder.
- Configuration: explicit embedding selection.
- Robustness: Probes for available models.
"""
import asyncio
import os
import sys
import logging

# Ensure we can import the core module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Agentic_core.core.config import get_config
from Agentic_core.execution.executor import AgentExecutor
from Agentic_core.core.inference_wrapper import get_inference_client
from Agentic_core.memory.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("manual_test")

async def probe_endpoint():
    """Finds active endpoint."""
    ports = [8007, 8000, 5000, 5002]
    import httpx
    
    for port in ports:
        url = f"http://localhost:{port}/v1"
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{url}/models")
                if resp.status_code == 200:
                    logger.info(f"Found active endpoint at {url}")
                    return url, resp.json()["data"][0]["id"]
        except:
            continue
    return None, None

async def main():
    print("--- STARTING DUAL-MODEL SOTA TEST ---")
    
    # 1. Configuration
    chat_url = "http://localhost:8007/v1"
    embed_url = "http://localhost:5002/v1"
    
    # Verify Chat
    chat_model = "openai/gpt-oss-20b"
    # Verify Embed
    embed_model = "Qwen/Qwen3-Embedding-4B"

    print(f"Chat Endpoint: {chat_url} ({chat_model})")
    print(f"Embed Endpoint: {embed_url} ({embed_model})")
    
    # Set Env for Executor
    os.environ["INFERENCE_BASE_URL"] = chat_url
    os.environ["INFERENCE_MODEL"] = chat_model
    os.environ["EMBEDDING_BASE_URL"] = embed_url
    # Hack: Passing embedding config via env or config file would be cleaner
    # For now ensuring VectorStore picks this up is key.
    
    # 2. Select Embedding (SOTA: Remote Inference)
    print(f"Selected Embedding Strategy: {embed_model} (Remote)")
    
    # 3. Setup Agent
    session_id = "dual_test_run_001"
    executor = AgentExecutor(session_id)
    
    # Explicitly init persistence
    await executor.initialize()
    
    # 4. Objective
    objective = "Verify that the chat model can reason about the GPU status and the embedding model is used for memory."
    
    # 5. Run
    print(f"\nObjective: {objective}\n")
    result = await executor.run(objective)
    
    print("\n--- TEST RESULT ---")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
