#!/usr/bin/env python3
"""
Quick test to verify GPU connections for benchmarking
"""

import sys
import logging
from llm_client import LMStudioClient
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gpu_connection(host, gpu_name):
    """Test connection to a specific GPU"""
    print(f"\n🧪 Testing {gpu_name} at {host}")
    print("-" * 50)
    
    try:
        # Override config
        original_host = config.lm_studio_host
        config.lm_studio_host = host
        
        # Test connection
        client = LMStudioClient()
        print(f"✅ Successfully connected to {gpu_name}")
        
        # Test a simple query
        test_messages = [
            {"role": "user", "content": "What is the capital of France? Answer in one word."}
        ]
        
        print("🔄 Testing simple query...")
        response = client.generate_response(test_messages, max_tokens=10)
        
        if response:
            print(f"✅ Query successful: {response[:50]}...")
        else:
            print("⚠️ Query returned empty response")
        
        # Restore original config
        config.lm_studio_host = original_host
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        # Restore original config
        config.lm_studio_host = original_host
        return False

def main():
    """Test both GPUs"""
    print("🚀 GPU Connection Test")
    print("=" * 60)
    
    gpus = [
        ("http://192.168.2.180:1234", "RTX 3080"),
        ("http://192.168.2.64:1234", "RTX 4090")
    ]
    
    results = {}
    for host, gpu_name in gpus:
        results[gpu_name] = test_gpu_connection(host, gpu_name)
    
    print(f"\n📊 Connection Test Results:")
    print("=" * 30)
    for gpu_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{gpu_name}: {status}")
    
    if all(results.values()):
        print("\n🎉 All GPUs are ready for benchmarking!")
        return 0
    else:
        print("\n⚠️ Some GPUs failed connection test")
        return 1

if __name__ == "__main__":
    sys.exit(main())