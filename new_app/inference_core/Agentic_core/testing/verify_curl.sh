#!/bin/bash
# Verify SOTA Inference Endpoints via CLI/cURL

echo "--- 1. Testing Chat Endpoint (gpt-oss-20b) ---"
echo "URL: http://localhost:8007/v1/chat/completions"

curl -s -X POST "http://localhost:8007/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
           "model": "openai/gpt-oss-20b",
           "messages": [{"role": "user", "content": "Hello! Status check."}],
           "max_tokens": 50
         }' | jq .
echo -e "\n--------------------------------------------\n"

echo "--- 2. Testing Embedding Endpoint - Normal ---"
echo "URL: http://localhost:8009/v1/embeddings"

curl -s -X POST "http://localhost:8009/v1/embeddings" \
     -H "Content-Type: application/json" \
     -d '{
           "model": "Qwen/Qwen3-Embedding-4B",
           "input": "Agentic Framework Single Vector"
         }' | jq '.data[0].embedding | length'
echo -e "\n--------------------------------------------\n"

echo "--- 3. Testing Embedding Endpoint - BATCHING (Concurrent) ---"
# Simulating client-side batching by sending a Request with a List of inputs
# vLLM handles this by batching internally.

curl -s -X POST "http://localhost:8009/v1/embeddings" \
     -H "Content-Type: application/json" \
     -d '{
           "model": "Qwen/Qwen3-Embedding-4B",
           "input": [
             "Batch Item 1: Hardware Check",
             "Batch Item 2: Software Check",
             "Batch Item 3: Memory Check"
           ]
         }' | jq '.data | length'

echo -e "\n(Should return 3 embeddings)"
echo -e "\n--------------------------------------------\n"
