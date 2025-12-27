"""
CLI API Tester for SOTA Agentic Framework.

Tests all inference core capabilities:
1. List Models
2. Chat Completion (Sanity Check)
3. Embedding Generation
"""
import asyncio
import os
import sys
import argparse
import aiohttp
import logging

# Append path to import core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from Agentic_core.core.inference_wrapper import get_inference_client
from Agentic_core.core.config import get_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CLI_TESTER")

async def test_list_models(base_url, name):
    logger.info(f"Testing List Models on {name} ({base_url})...")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{base_url}/models") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [m['id'] for m in data['data']]
                    logger.info(f"✅ PASSED. Found models: {models}")
                    return True
                else:
                    logger.error(f"❌ FAILED. Status: {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ CONNECTION FAILED: {e}")
            return False

async def test_chat(base_url, model):
    logger.info(f"Testing Chat Completion on {base_url} with {model}...")
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say 'System Operational' if you hear me."}],
        "max_tokens": 20
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{base_url}/chat/completions", json=payload, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data['choices'][0]['message'].get('content')
                    reasoning = data['choices'][0]['message'].get('reasoning_content')
                    
                    if content:
                        logger.info(f"✅ PASSED. Response: {content.strip()[:50]}...")
                    elif reasoning:
                        logger.info(f"✅ PASSED. (Reasoning Only): {reasoning.strip()[:50]}...")
                    else:
                        logger.warning("⚠️ PASSED (Empty Response). Check max_tokens?")
                    return True
                else:
                    text = await resp.text()
                    logger.error(f"❌ FAILED. Status: {resp.status} - {text}")
                    return False
        except Exception as e:
            logger.error(f"❌ ERROR: {e}")
            return False

async def test_embedding(base_url, model):
    logger.info(f"Testing Embeddings on {base_url} with {model}...")
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "input": "Agentic Framework SOTA Test"
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{base_url}/embeddings", json=payload, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    embed = data['data'][0]['embedding']
                    dim = len(embed)
                    logger.info(f"✅ PASSED. Embedding Dimension: {dim}")
                    return True
                else:
                    text = await resp.text()
                    logger.error(f"❌ FAILED. Status: {resp.status} - {text}")
                    return False
        except Exception as e:
            logger.error(f"❌ ERROR: {e}")
            return False

async def main():
    print("--- SOTA CLI API TESTER ---")
    
    # Configuration
    chat_url = "http://localhost:8007/v1"
    chat_model = "openai/gpt-oss-20b"
    
    embed_url = "http://localhost:8009/v1"
    embed_model = "Qwen/Qwen3-Embedding-4B"
    
    results = []
    
    # 1. Chat Node Tests
    print("\n[Node 1: CHAT (GPUs 0-3)]")
    if await test_list_models(chat_url, "Chat Node"):
        results.append(await test_chat(chat_url, chat_model))
    else:
        results.append(False)
        
    # 2. Embed Node Tests
    print("\n[Node 2: EMBED (GPUs 4-5)]")
    if await test_list_models(embed_url, "Embed Node"):
        results.append(await test_embedding(embed_url, embed_model))
    else:
        results.append(False)
        
    if all(results):
        print("\n🎉 ALL TESTS PASSED. Infrastructure is SOTA-Ready.")
        sys.exit(0)
    else:
        print("\n💥 ONE OR MORE TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
