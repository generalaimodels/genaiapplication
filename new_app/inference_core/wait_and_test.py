"""
Verification Wrapper.

Polls inference endpoints for readiness before launching the test.
"""
import time
import requests
import subprocess
import sys

def wait_for_endpoint(url, name, retries=60, sleep=10):
    print(f"Waiting for {name} at {url}...")
    for i in range(retries):
        try:
            resp = requests.get(f"{url}/models")
            if resp.status_code == 200:
                print(f"✅ {name} is READY!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        print(f"[{i+1}/{retries}] {name} loading... (sleeping {sleep}s)")
        time.sleep(sleep)
    return False

if __name__ == "__main__":
    chat_ready = wait_for_endpoint("http://localhost:8007/v1", "Chat (gpt-oss-20b)")
    embed_ready = wait_for_endpoint("http://localhost:5002/v1", "Embed (Qwen3-4B)")
    
    if chat_ready and embed_ready:
        print("\n🚀 All Systems GO. Launching SOTA CLI Verification...\n")
        subprocess.run([sys.executable, "Agentic_core/testing/test_all_api.py"], check=False)
        print("\n📜 Launching cURL Verification...\n")
        subprocess.run(["bash", "Agentic_core/testing/verify_curl.sh"], check=False)
    else:
        print("\n❌ Timed out waiting for models to load. Please check Docker logs.")
