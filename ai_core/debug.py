# Debug script
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clients.litellm_client import LiteLLMClient
from clients.base_client import ChatMessage

client = LiteLLMClient(
    model='gemini-2.0-flash',
    api_key='AIzaSyB2V6xuoi3eVdg-8g1Xqm9W8AubFw5Z0_0'
)

print(f"Model string: {client._get_model_string()}")
print(f"API Key set: {bool(client.api_key)}")
print(f"API Key: {client.api_key[:10]}...")

# Test build kwargs
messages = [ChatMessage(role="user", content="Hi")]
formatted = client._prepare_messages(messages)
kwargs = client._build_completion_kwargs(formatted, None, False)
print(f"\nCompletion kwargs:")
for k, v in kwargs.items():
    if k == 'api_key':
        print(f"  {k}: {v[:10]}...")
    else:
        print(f"  {k}: {v}")

# Now try the actual call
print("\n--- Trying actual call ---")
try:
    response = client.chat(messages)
    print(f"SUCCESS: {response.content}")
except Exception as e:
    print(f"ERROR: {e}")
