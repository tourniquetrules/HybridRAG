#!/usr/bin/env python3
"""
Quick Vector DB Benchmark
Simplified benchmarking tool for rapid testing of vector database performance
"""

import time
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import argparse

from hybrid_rag import HybridRAGSystem

class QuickBenchmark:
    def __init__(self):
        self.rag_system = HybridRAGSystem()
        
    def single_query_test(self, query: str) -> dict:
        """Test a single query and return timing breakdown"""
        start_time = time.time()
        
                # Time the embedding
        embed_start = time.time()
        embedding = self.rag_system.vector_db.embedding_model.encode([query])
        embed_time = time.time() - embed_start
        
        # Time search
        search_start = time.time()
        results = self.rag_system.vector_db.collection.query(
            query_embeddings=embedding,
            n_results=5,
            include=['documents', 'metadatas', 'distances']
        )
        search_time = time.time() - search_start
        
        total_time = time.time() - start_time
        
        return {
            'embedding_time': embed_time,
            'search_time': search_time,
            'total_time': total_time,
            'results_count': len(results['documents'][0]) if results['documents'] else 0
        }
    
    def concurrent_test(self, num_users: int, query: str = "What is sepsis management?"):
        """Test concurrent users with the same query"""
        print(f"\n🔥 Testing {num_users} concurrent users...")
        
        start_time = time.time()
        results = []
        cpu_before = psutil.cpu_percent()
        mem_before = psutil.virtual_memory().percent
        
        with ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [executor.submit(self.single_query_test, query) for _ in range(num_users)]
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=10)
                    results.append(result)
                except Exception as e:
                    print(f"Query failed: {e}")
        
        duration = time.time() - start_time
        cpu_after = psutil.cpu_percent()
        mem_after = psutil.virtual_memory().percent
        
        # Calculate statistics
        if results:
            total_times = [r['total_time'] for r in results]
            embedding_times = [r['embedding_time'] for r in results]
            search_times = [r['search_time'] for r in results]
            
            print(f"✅ Results for {num_users} concurrent users:")
            print(f"  Success Rate:     {len(results)}/{num_users} ({len(results)/num_users*100:.1f}%)")
            print(f"  Total Duration:   {duration:.2f}s")
            print(f"  Queries/Second:   {len(results)/duration:.2f}")
            print(f"  Avg Total Time:   {statistics.mean(total_times)*1000:.1f}ms")
            print(f"  Avg Embedding:    {statistics.mean(embedding_times)*1000:.1f}ms")
            print(f"  Avg Search:       {statistics.mean(search_times)*1000:.1f}ms")
            print(f"  P95 Time:         {sorted(total_times)[int(len(total_times)*0.95)]*1000:.1f}ms")
            print(f"  CPU Usage:        {cpu_before:.1f}% → {cpu_after:.1f}%")
            print(f"  Memory Usage:     {mem_before:.1f}% → {mem_after:.1f}%")
        else:
            print("❌ All queries failed!")

def main():
    parser = argparse.ArgumentParser(description='Quick Vector DB Benchmark')
    parser.add_argument('--users', '-u', type=int, nargs='+', 
                       default=[1, 5, 10, 20, 50], 
                       help='Number of concurrent users to test')
    parser.add_argument('--query', '-q', type=str, 
                       default="What is the management of septic shock?",
                       help='Query to test with')
    
    args = parser.parse_args()
    
    print("🚀 Quick Vector Database Benchmark")
    print("="*50)
    
    benchmark = QuickBenchmark()
    
    # Test single query first
    print("\n📊 Single Query Baseline:")
    result = benchmark.single_query_test(args.query)
    print(f"  Embedding Time:   {result['embedding_time']*1000:.1f}ms")
    print(f"  Search Time:      {result['search_time']*1000:.1f}ms")
    print(f"  Total Time:       {result['total_time']*1000:.1f}ms")
    print(f"  Results Found:    {result['results_count']}")
    
    # Test concurrent users
    for num_users in args.users:
        benchmark.concurrent_test(num_users, args.query)
        time.sleep(1)  # Brief pause between tests
    
    print("\n✨ Benchmark complete!")

if __name__ == "__main__":
    main()