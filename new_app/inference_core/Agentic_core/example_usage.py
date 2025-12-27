"""
Verification Script for Agentic Framework 3.0.
"""
import asyncio
import os
import sys
import logging

# Ensure the package is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Setup Environment for Inference Core (Pointing to mock or local vllm)
os.environ["INFERENCE_BASE_URL"] = "http://localhost:8000/v1"
os.environ["INFERENCE_API_KEY"] = "EMPTY" 
os.environ["STORAGE_PATH"] = "agent_test.db"

# Configure Logging
logging.basicConfig(level=logging.INFO)

async def main():
    print("🚀 Initializing Agentic Framework 3.0...")
    
    try:
        from inference_core.Agentic_core.execution.executor import AgentExecutor
    except ImportError as e:
        print(f"Import Error: {e}")
        print("Please ensure you are running this from the correct directory.")
        return

    session_id = "test_session_001"
    agent = AgentExecutor(session_id)
    
    print("💾 Initializing Storage...")
    await agent.initialize()
    
    objective = "Write a haiku about code optimization."
    print(f"🎯 Objective: {objective}")
    
    print("⚡ Running Agent...")
    try:
        result = await agent.run(objective)
        print("\n✅ Result:\n" + "="*40)
        print(result)
        print("="*40)
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
