# =============================================================================
# EXAMPLE USAGE - Advanced Chatbot Core AI
# =============================================================================
# Comprehensive examples demonstrating all features of the AI Core system.
# =============================================================================

"""
ADVANCED CHATBOT CORE AI - Usage Examples

This file demonstrates how to use the ChatbotCoreAI system with various
LLM providers including OpenAI, Anthropic, Gemini, VLLM, and more.

Run examples:
    python examples.py

Note: Set your API keys via environment variables or pass directly.
"""

import asyncio
from pathlib import Path


# =============================================================================
# EXAMPLE 1: OPENAI BASIC USAGE
# =============================================================================

def example_openai_basic():
    """Basic OpenAI usage."""
    print("\n" + "="*60)
    print("EXAMPLE 1: OpenAI Basic Usage")
    print("="*60)
    
    from AI_core import ChatbotCoreAI
    
    # Initialize with OpenAI
    chatbot = ChatbotCoreAI(
        model="gpt-4o-mini",
        api_key="your-openai-api-key",  # Or set OPENAI_API_KEY env var
        temperature=0.7
    )
    
    # Simple chat
    response = chatbot.chat("What is Python programming language?")
    print(f"\nResponse: {response.content[:500]}...")
    print(f"Tokens used: {response.usage}")


# =============================================================================
# EXAMPLE 2: VLLM SELF-HOSTED
# =============================================================================

def example_vllm_selfhosted():
    """VLLM self-hosted server usage."""
    print("\n" + "="*60)
    print("EXAMPLE 2: VLLM Self-Hosted")
    print("="*60)
    
    from AI_core import ChatbotCoreAI
    
    # Connect to VLLM server
    chatbot = ChatbotCoreAI(
        model="your-model-name",
        base_url="http://10.180.93.12:8007/v1",  # Your VLLM endpoint
        api_key="EMPTY",  # VLLM typically uses "EMPTY"
        temperature=0.7,
        max_tokens=2048
    )
    
    # Chat
    response = chatbot.chat(
        "Explain machine learning in simple terms",
        max_tokens=500
    )
    print(f"\nResponse: {response.content}")


# =============================================================================
# EXAMPLE 3: GEMINI
# =============================================================================

def example_gemini():
    """Google Gemini usage."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Google Gemini")
    print("="*60)
    
    from AI_core import ChatbotCoreAI
    
    chatbot = ChatbotCoreAI(
        model="gemini-3-flash-preview",
        provider="gemini",
        api_key="AIzaSyB2V6xuoi3eVdg-8g1Xqm9W8AubFw5Z0_0",  # Or set GEMINI_API_KEY
        temperature=0.7
    )
    
    response = chatbot.chat("What are the benefits of cloud computing?")
    print(f"\nResponse: {response.content[:500]}...")


# =============================================================================
# EXAMPLE 4: ANTHROPIC CLAUDE
# =============================================================================

def example_anthropic():
    """Anthropic Claude usage."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Anthropic Claude")
    print("="*60)
    
    from AI_core import ChatbotCoreAI
    
    chatbot = ChatbotCoreAI(
        model="claude-3-5-sonnet-20241022",
        provider="anthropic",
        api_key="your-anthropic-api-key",  # Or set ANTHROPIC_API_KEY
    )
    
    response = chatbot.chat(
        "Write a haiku about programming",
        temperature=0.9
    )
    print(f"\nResponse:\n{response.content}")


# =============================================================================
# EXAMPLE 5: SESSION-BASED CONVERSATION
# =============================================================================

def example_session_conversation():
    """Multi-turn conversation with session."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Session-Based Conversation")
    print("="*60)
    
    from AI_core import ChatbotCoreAI
    
    chatbot = ChatbotCoreAI(
        model="gpt-4o-mini",
        api_key="your-api-key"
    )
    
    # Create a session
    session = chatbot.create_session(
        user_id="demo-user",
        system_prompt="You are a helpful Python tutor."
    )
    
    print(f"Session created: {session.session_id}")
    
    # Multi-turn conversation
    conversations = [
        "I want to learn Python",
        "What should I start with?",
        "Can you show me a simple example?",
    ]
    
    for query in conversations:
        print(f"\nUser: {query}")
        response = chatbot.chat(query, session_id=session.session_id)
        print(f"Assistant: {response.content[:300]}...")
    
    # Get conversation history
    history = chatbot.get_history(session.session_id)
    print(f"\nTotal messages in history: {len(history)}")


# =============================================================================
# EXAMPLE 6: DOCUMENT-AUGMENTED CHAT (RAG)
# =============================================================================

def example_rag():
    """Document-augmented responses (RAG)."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Document-Augmented Chat (RAG)")
    print("="*60)
    
    from AI_core import ChatbotCoreAI
    
    chatbot = ChatbotCoreAI(
        model="gpt-4o-mini",
        api_key="your-api-key"
    )
    
    # Sample documents (can also use file paths)
    documents = [
        """
        Python is a high-level, interpreted programming language known for 
        its readability and simplicity. It was created by Guido van Rossum 
        and first released in 1991. Python supports multiple programming 
        paradigms including procedural, object-oriented, and functional.
        """,
        """
        Key features of Python include:
        - Dynamic typing and automatic memory management
        - Large standard library ("batteries included")
        - Support for modules and packages
        - Easy integration with C/C++ for performance
        - Active community and extensive third-party packages via PyPI
        """
    ]
    
    response = chatbot.chat_with_documents(
        query="What are Python's key features and who created it?",
        documents=documents,
        top_k=5
    )
    
    print(f"\nResponse: {response.content}")
    print(f"Context chunks used: {response.metadata.get('context_chunks', 'N/A')}")


# =============================================================================
# EXAMPLE 7: STREAMING RESPONSES
# =============================================================================

def example_streaming():
    """Real-time streaming responses."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Streaming Responses")
    print("="*60)
    
    from AI_core import ChatbotCoreAI
    
    chatbot = ChatbotCoreAI(
        model="gpt-4o-mini",
        api_key="your-api-key"
    )
    
    print("\nStreaming response:")
    print("-" * 40)
    
    for chunk in chatbot.stream_chat(
        "Write a short poem about coding",
        temperature=0.9
    ):
        print(chunk.content, end="", flush=True)
    
    print("\n" + "-" * 40)


# =============================================================================
# EXAMPLE 8: ASYNC STREAMING
# =============================================================================

async def example_async_streaming():
    """Async streaming responses."""
    print("\n" + "="*60)
    print("EXAMPLE 8: Async Streaming")
    print("="*60)
    
    from AI_core import ChatbotCoreAI
    
    chatbot = ChatbotCoreAI(
        model="gpt-4o-mini",
        api_key="your-api-key"
    )
    
    print("\nAsync streaming response:")
    print("-" * 40)
    
    async for chunk in chatbot.astream_chat("Explain quantum computing briefly"):
        print(chunk.content, end="", flush=True)
    
    print("\n" + "-" * 40)


# =============================================================================
# EXAMPLE 9: BATCH PROCESSING
# =============================================================================

async def example_batch_processing():
    """Process multiple queries concurrently."""
    print("\n" + "="*60)
    print("EXAMPLE 9: Batch Processing")
    print("="*60)
    
    from AI_core import ChatbotCoreAI
    
    chatbot = ChatbotCoreAI(
        model="gpt-4o-mini",
        api_key="your-api-key"
    )
    
    queries = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
        "What is Go?",
    ]
    
    print(f"\nProcessing {len(queries)} queries concurrently...")
    
    responses = await chatbot.abatch_chat(
        queries,
        max_concurrent=3
    )
    
    for query, response in zip(queries, responses):
        print(f"\nQ: {query}")
        print(f"A: {response.content[:150]}...")


# =============================================================================
# EXAMPLE 10: FEEDBACK COLLECTION
# =============================================================================

def example_feedback():
    """Collect user feedback."""
    print("\n" + "="*60)
    print("EXAMPLE 10: Feedback Collection")
    print("="*60)
    
    from AI_core import ChatbotCoreAI
    
    chatbot = ChatbotCoreAI(
        model="gpt-4o-mini",
        api_key="your-api-key"
    )
    
    # Create session and chat
    session = chatbot.create_session()
    response = chatbot.chat("What is AI?", session_id=session.session_id)
    
    # Submit feedback
    chatbot.submit_feedback(
        session_id=session.session_id,
        feedback_type="thumbs_up"
    )
    
    chatbot.submit_feedback(
        session_id=session.session_id,
        feedback_type="rating",
        value=5,
        comment="Very helpful response!"
    )
    
    print("Feedback submitted successfully!")


# =============================================================================
# EXAMPLE 11: PROVIDER LISTING
# =============================================================================

def example_list_providers():
    """List available providers and models."""
    print("\n" + "="*60)
    print("EXAMPLE 11: Available Providers")
    print("="*60)
    
    from AI_core import ChatbotCoreAI
    
    providers = ChatbotCoreAI.list_providers()
    print(f"\nSupported Providers ({len(providers)}):")
    
    for provider in providers:
        models = ChatbotCoreAI.get_provider_models(provider)
        print(f"\n  {provider}:")
        for model in models[:3]:  # Show first 3 models
            print(f"    - {model}")
        if len(models) > 3:
            print(f"    - ... and {len(models)-3} more")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all examples."""
    print("="*60)
    print("ADVANCED CHATBOT CORE AI - Examples")
    print("="*60)
    print("\nNote: Update API keys before running examples.")
    print("Set via environment variables or pass directly.")
    
    # Run synchronous examples
    # Uncomment the examples you want to run:
    
    # example_openai_basic()
    # example_vllm_selfhosted()
    example_gemini()
    # example_anthropic()
    # example_session_conversation()
    # example_rag()
    # example_streaming()
    # example_feedback()
    # example_list_providers()
    
    # Run async examples
    # asyncio.run(example_async_streaming())
    # asyncio.run(example_batch_processing())
    
    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)


if __name__ == "__main__":
    main()
