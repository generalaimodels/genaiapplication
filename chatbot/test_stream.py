#!/usr/bin/env python3
"""Test script to debug OpenAI client streaming with vLLM."""

import asyncio
from openai import AsyncOpenAI

async def test_stream():
    client = AsyncOpenAI(
        base_url="http://localhost:8007/v1",
        api_key="dummy"
    )
    
    print("Starting stream test...")
    
    stream = await client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=200,
        stream=True
    )
    
    token_count = 0
    async for chunk in stream:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            content = getattr(delta, 'content', None)
            reasoning = getattr(delta, 'reasoning_content', None)
            
            if token_count < 10:
                print(f"Chunk {token_count}:")
                print(f"  content: {repr(content)}")
                print(f"  reasoning_content: {repr(reasoning)}")
                print(f"  delta attrs: {dir(delta)}")
            
            if content:
                print(f"  >>> CONTENT TOKEN: {content}")
            if reasoning:
                print(f"  >>> REASONING TOKEN: {reasoning}")
            
            token_count += 1
    
    print(f"\nTotal chunks: {token_count}")

if __name__ == "__main__":
    asyncio.run(test_stream())
