#!/usr/bin/env python3
"""
Comprehensive RAG System Benchmarking Tool
Tests the performance of both vector database and LLM inference:
1. Question embedding generation
2. Vector similarity search
3. LLM inference on RTX 3080 (192.168.2.180:1234)
4. End-to-end query processing
5. Concurrent user simulation
6. Performance metrics collection
"""

import asyncio
import time
import statistics
import json
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
import threading
from dataclasses import dataclass
import psutil
import numpy as np
from pathlib import Path
import requests

from hybrid_rag import HybridRAGSystem
from llm_client import LMStudioClient
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Single benchmark test result"""
    user_id: int
    query: str
    embedding_time: float
    search_time: float
    llm_inference_time: float
    total_time: float
    num_results: int
    response_length: int
    success: bool
    error: str = ""

@dataclass
class BenchmarkSummary:
    """Overall benchmark summary"""
    total_queries: int
    successful_queries: int
    failed_queries: int
    concurrent_users: int
    total_duration: float
    avg_embedding_time: float
    avg_search_time: float
    avg_llm_inference_time: float
    avg_total_time: float
    min_time: float
    max_time: float
    p50_time: float
    p95_time: float
    p99_time: float
    queries_per_second: float
    cpu_usage: float
    memory_usage: float
    errors: List[str]

class ComprehensiveRAGBenchmark:
    """Comprehensive RAG System Benchmark Tool"""
    
    def __init__(self, llm_host_override=None):
        """Initialize benchmark with optional LLM host override for GPU comparison"""
        self.original_lm_studio_host = None
        
        # Override LM Studio host if specified (for GPU comparison)
        if llm_host_override:
            self.original_lm_studio_host = config.lm_studio_host
            config.lm_studio_host = llm_host_override
            print(f"🔄 Overriding LM Studio host to: {llm_host_override}")
        
        self.rag_system = HybridRAGSystem()
        
        # If we overrode the host, we need to update the RAG system's LLM client too
        if llm_host_override:
            print(f"🔄 Updating RAG system's LLM client to use: {llm_host_override}")
            self.rag_system.llm_client = LMStudioClient()
        
        self.test_queries = [
            "How did ChatGPT perform on the Emergency Medicine boards?",
            "What is the effect of hyperoxia on endovascular therapy for stroke?",
            "What are the criteria for early discharge for drowning patients?",
            "Does skin glue reduce intravenous catheter failure in children?",
            "How accurate were fundus photos for triaging vision loss?"
        ]
    
    async def initialize_system(self):
        """Initialize the RAG system and LLM client"""
        logger.info("Initializing RAG system for benchmarking...")
        try:
            self.rag_system = HybridRAGSystem()
            if not self.rag_system.vector_db:
                raise Exception("Vector database not initialized")
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
        
        logger.info("Initializing LLM client for inference benchmarking...")
        try:
            # Allow override of LM Studio host for benchmarking different GPUs
            if hasattr(self, 'llm_host_override') and self.llm_host_override:
                original_host = config.lm_studio_host
                config.lm_studio_host = self.llm_host_override
                logger.info(f"Using LLM host override: {self.llm_host_override}")
                
                self.llm_client = LMStudioClient()
                logger.info(f"LLM client initialized successfully for {config.lm_studio_host}")
                
                # Restore original host
                config.lm_studio_host = original_host
            else:
                self.llm_client = LMStudioClient()
                logger.info(f"LLM client initialized successfully for {config.lm_studio_host}")
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise
    
    def benchmark_single_query(self, user_id: int, query: str, include_llm: bool = True) -> BenchmarkResult:
        """Benchmark a single query through the complete RAG pipeline"""
        start_time = time.time()
        
        try:
            # Time embedding generation and vector search using HybridRAG
            embedding_start = time.time()
            search_results = self.rag_system.search(query, max_results=5, use_knowledge_graph=False)  # Disable KG for faster benchmarking
            search_time = time.time() - embedding_start
            
            # Convert to expected format for compatibility
            results = {
                'documents': [[result.get('content', result.get('text', '')) for result in search_results]],
                'metadatas': [[result.get('metadata', {}) for result in search_results]],
                'distances': [[1.0 - result.get('score', result.get('similarity_score', 0)) for result in search_results]]  # Convert similarity to distance
            }
            embedding_time = search_time * 0.3  # Estimate embedding portion
            
            # Time LLM inference (if enabled)
            llm_inference_time = 0
            response_length = 0
            
            if include_llm and self.llm_client:
                llm_start = time.time()
                
                # Prepare context from retrieved documents
                context_docs = []
                if results['documents'] and results['documents'][0]:
                    raw_docs = results['documents'][0][:3]  # Use top 3 results
                    metadatas = results.get('metadatas', [[{}] * len(raw_docs)])[0]
                    for i, doc in enumerate(raw_docs):
                        context_docs.append({
                            'text': doc,  # Fixed: use 'text' instead of 'content'
                            'metadata': metadatas[i] if i < len(metadatas) else {}
                        })
                
                # Generate LLM response
                try:
                    response = self.llm_client.generate_rag_response(query, context_docs)
                    response_length = len(response) if response else 0
                except Exception as llm_error:
                    logger.warning(f"LLM inference failed for user {user_id}: {llm_error}")
                    response_length = 0
                
                llm_inference_time = time.time() - llm_start
            
            total_time = time.time() - start_time
            num_results = len(results['documents'][0]) if results['documents'] else 0
            
            return BenchmarkResult(
                user_id=user_id,
                query=query,
                embedding_time=embedding_time,
                search_time=search_time,
                llm_inference_time=llm_inference_time,
                total_time=total_time,
                num_results=num_results,
                response_length=response_length,
                success=True
            )
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Query failed for user {user_id}: {e}")
            
            return BenchmarkResult(
                user_id=user_id,
                query=query,
                embedding_time=0,
                search_time=0,
                llm_inference_time=0,
                total_time=total_time,
                num_results=0,
                response_length=0,
                success=False,
                error=str(e)
            )
    
    def collect_system_metrics(self):
        """Collect system performance metrics"""
        return {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3)
        }
    
    def run_concurrent_benchmark(self, num_users: int, queries_per_user: int = 5, include_llm: bool = True) -> List[BenchmarkResult]:
        """Run concurrent benchmark with multiple simulated users"""
        llm_mode = "with LLM inference" if include_llm else "vector DB only"
        logger.info(f"Starting concurrent benchmark ({llm_mode}): {num_users} users, {queries_per_user} queries each")
        
        results = []
        start_time = time.time()
        
        # Start system monitoring
        monitoring_stop = threading.Event()
        monitoring_thread = threading.Thread(
            target=self._monitor_system_metrics, 
            args=(monitoring_stop,)
        )
        monitoring_thread.start()
        
        # Create thread pool for concurrent execution
        with ThreadPoolExecutor(max_workers=num_users) as executor:
            # Submit all tasks
            futures = []
            for user_id in range(num_users):
                for query_num in range(queries_per_user):
                    query = np.random.choice(self.test_queries)
                    future = executor.submit(self.benchmark_single_query, user_id, query, include_llm)
                    futures.append(future)
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=60)  # 60 second timeout per query (increased for LLM)
                    results.append(result)
                    
                    # Log progress
                    if len(results) % 10 == 0:
                        logger.info(f"Completed {len(results)}/{len(futures)} queries")
                        
                except Exception as e:
                    logger.error(f"Query execution failed: {e}")
        
        # Stop monitoring
        monitoring_stop.set()
        monitoring_thread.join()
        
        duration = time.time() - start_time
        logger.info(f"Benchmark completed in {duration:.2f} seconds")
        
        return results
    
    def _monitor_system_metrics(self, stop_event):
        """Monitor system metrics during benchmark"""
        while not stop_event.is_set():
            metrics = self.collect_system_metrics()
            self.system_metrics.append(metrics)
            time.sleep(0.5)  # Collect metrics every 500ms
    
    def analyze_results(self, results: List[BenchmarkResult], concurrent_users: int) -> BenchmarkSummary:
        """Analyze benchmark results and generate summary"""
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        if not successful_results:
            return BenchmarkSummary(
                total_queries=len(results),
                successful_queries=0,
                failed_queries=len(failed_results),
                concurrent_users=concurrent_users,
                total_duration=0,
                avg_embedding_time=0,
                avg_search_time=0,
                avg_llm_inference_time=0,
                avg_total_time=0,
                min_time=0,
                max_time=0,
                p50_time=0,
                p95_time=0,
                p99_time=0,
                queries_per_second=0,
                cpu_usage=0,
                memory_usage=0,
                errors=[r.error for r in failed_results]
            )
        
        # Calculate timing statistics
        total_times = [r.total_time for r in successful_results]
        embedding_times = [r.embedding_time for r in successful_results]
        search_times = [r.search_time for r in successful_results]
        llm_inference_times = [r.llm_inference_time for r in successful_results]
        
        # Calculate system metrics
        avg_cpu = statistics.mean([m['cpu_percent'] for m in self.system_metrics]) if self.system_metrics else 0
        avg_memory = statistics.mean([m['memory_percent'] for m in self.system_metrics]) if self.system_metrics else 0
        
        # Calculate total duration from first to last successful query
        if successful_results:
            # Estimate total duration based on timing
            total_duration = max(total_times) if total_times else 0
        else:
            total_duration = 0
        
        return BenchmarkSummary(
            total_queries=len(results),
            successful_queries=len(successful_results),
            failed_queries=len(failed_results),
            concurrent_users=concurrent_users,
            total_duration=total_duration,
            avg_embedding_time=statistics.mean(embedding_times),
            avg_search_time=statistics.mean(search_times),
            avg_llm_inference_time=statistics.mean(llm_inference_times),
            avg_total_time=statistics.mean(total_times),
            min_time=min(total_times),
            max_time=max(total_times),
            p50_time=statistics.median(total_times),
            p95_time=np.percentile(total_times, 95),
            p99_time=np.percentile(total_times, 99),
            queries_per_second=len(successful_results) / max(total_duration, 0.001),  # Avoid division by zero
            cpu_usage=avg_cpu,
            memory_usage=avg_memory,
            errors=[r.error for r in failed_results if r.error]
        )
    
    def print_results(self, summary: BenchmarkSummary, title: str = "RAG SYSTEM BENCHMARK RESULTS"):
        """Print benchmark results in a formatted way"""
        print("\n" + "="*80)
        print(title)
        print("="*80)
        
        print(f"\n📊 OVERVIEW:")
        print(f"  Total Queries:        {summary.total_queries}")
        print(f"  Successful:           {summary.successful_queries} ({summary.successful_queries/summary.total_queries*100:.1f}%)")
        print(f"  Failed:               {summary.failed_queries} ({summary.failed_queries/summary.total_queries*100:.1f}%)")
        print(f"  Concurrent Users:     {summary.concurrent_users}")
        print(f"  Duration:             {summary.total_duration:.2f}s")
        
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"  Queries/Second:       {summary.queries_per_second:.2f}")
        print(f"  Avg Total Time:       {summary.avg_total_time*1000:.2f}ms")
        print(f"  Avg Embedding Time:   {summary.avg_embedding_time*1000:.2f}ms")
        print(f"  Avg Search Time:      {summary.avg_search_time*1000:.2f}ms")
        if hasattr(summary, 'avg_llm_inference_time') and summary.avg_llm_inference_time > 0:
            print(f"  Avg LLM Inference:    {summary.avg_llm_inference_time*1000:.2f}ms")
        
        print(f"\n📈 LATENCY DISTRIBUTION:")
        print(f"  Min Time:             {summary.min_time*1000:.2f}ms")
        print(f"  P50 (Median):         {summary.p50_time*1000:.2f}ms")
        print(f"  P95:                  {summary.p95_time*1000:.2f}ms")
        print(f"  P99:                  {summary.p99_time*1000:.2f}ms")
        print(f"  Max Time:             {summary.max_time*1000:.2f}ms")
        
        print(f"\n🖥️  SYSTEM RESOURCES:")
        print(f"  Avg CPU Usage:        {summary.cpu_usage:.1f}%")
        print(f"  Avg Memory Usage:     {summary.memory_usage:.1f}%")
        
        if summary.errors:
            print(f"\n❌ ERRORS ({len(summary.errors)}):")
            for i, error in enumerate(summary.errors[:5]):  # Show first 5 errors
                print(f"  {i+1}. {error}")
            if len(summary.errors) > 5:
                print(f"  ... and {len(summary.errors)-5} more errors")
        
        print("\n" + "="*80)
    
    def print_summary(self, summary: BenchmarkSummary, title: str = "Benchmark Summary"):
        """Print a compact summary of benchmark results"""
        success_rate = summary.successful_queries / summary.total_queries * 100 if summary.total_queries > 0 else 0
        print(f"\n✅ {title}")
        print(f"   Success: {summary.successful_queries}/{summary.total_queries} ({success_rate:.1f}%)")
        print(f"   QPS: {summary.queries_per_second:.1f}")
        print(f"   Avg Response: {summary.avg_total_time*1000:.0f}ms")
        if hasattr(summary, 'avg_llm_inference_time') and summary.avg_llm_inference_time > 0:
            print(f"   (Embedding: {summary.avg_embedding_time*1000:.0f}ms, Search: {summary.avg_search_time*1000:.0f}ms, LLM: {summary.avg_llm_inference_time*1000:.0f}ms)")
        else:
            print(f"   (Embedding: {summary.avg_embedding_time*1000:.0f}ms, Search: {summary.avg_search_time*1000:.0f}ms)")
    
    def save_results(self, summary: BenchmarkSummary, results: List[BenchmarkResult], filename: str = None):
        """Save detailed results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        data = {
            'summary': {
                'total_queries': summary.total_queries,
                'successful_queries': summary.successful_queries,
                'failed_queries': summary.failed_queries,
                'concurrent_users': summary.concurrent_users,
                'total_duration': summary.total_duration,
                'avg_embedding_time': summary.avg_embedding_time,
                'avg_search_time': summary.avg_search_time,
                'avg_llm_inference_time': summary.avg_llm_inference_time,
                'avg_total_time': summary.avg_total_time,
                'min_time': summary.min_time,
                'max_time': summary.max_time,
                'p50_time': summary.p50_time,
                'p95_time': summary.p95_time,
                'p99_time': summary.p99_time,
                'queries_per_second': summary.queries_per_second,
                'cpu_usage': summary.cpu_usage,
                'memory_usage': summary.memory_usage,
                'errors': summary.errors
            },
            'detailed_results': [
                {
                    'user_id': r.user_id,
                    'query': r.query,
                    'embedding_time': r.embedding_time,
                    'search_time': r.search_time,
                    'llm_inference_time': r.llm_inference_time,
                    'total_time': r.total_time,
                    'num_results': r.num_results,
                    'response_length': r.response_length,
                    'success': r.success,
                    'error': r.error
                }
                for r in results
            ],
            'system_metrics': self.system_metrics,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'vector_db_path': config.vector_db_path,
                'collection_name': config.collection_name
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")

async def run_gpu_benchmark(gpu_host: str, gpu_name: str):
    """Run benchmark for a specific GPU"""
    print(f"\n" + "="*80)
    print(f"🎮 BENCHMARKING {gpu_name} at {gpu_host}")
    print("="*80)
    
    benchmark = ComprehensiveRAGBenchmark(llm_host_override=gpu_host)
    
    try:
        # Initialize system
        await benchmark.initialize_system()
        
        # Test scenarios with increasing concurrent users
        test_scenarios = [
            (1, 3),     # 1 user, 3 queries
            (3, 2),     # 3 users, 2 queries each
            (5, 2),     # 5 users, 2 queries each
        ]
        
        all_results = []
        
        # Run full RAG pipeline tests (with LLM inference)
        print(f"\n🤖 FULL RAG PIPELINE BENCHMARKS ({gpu_name})")
        print("="*60)
        

        
        for num_users, queries_per_user in test_scenarios:
            print(f"\n🚀 Testing {num_users} concurrent users with {queries_per_user} queries each...")
            
            # Reset metrics for this test
            benchmark.system_metrics = []
            
            # Run benchmark with LLM
            results = benchmark.run_concurrent_benchmark(num_users, queries_per_user, include_llm=True)
            
            # Analyze results
            summary = benchmark.analyze_results(results, num_users)
            benchmark.print_summary(summary, f"{gpu_name} - {num_users} Users")
            
            all_results.append({
                'scenario': f"{gpu_name.lower().replace(' ', '_')}_{num_users}_users",
                'summary': summary,
                'results': results
            })
        
        # Save individual GPU benchmark results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gpu_safe_name = gpu_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').replace('+', '_plus_')
        filename = f"individual_benchmark_{gpu_safe_name}_{timestamp}.json"
        
        benchmark_data = {
            'gpu_name': gpu_name,
            'gpu_host': gpu_host,
            'timestamp': datetime.now().isoformat(),
            'scenarios': all_results,
            'system_info': {
                'vector_db_path': config.vector_db_path,
                'collection_name': config.collection_name
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(benchmark_data, f, indent=2, default=str)
        
        print(f"\n💾 Individual benchmark results saved to: {filename}")
        
        return all_results
        
    except Exception as e:
        logger.error(f"GPU benchmark failed for {gpu_name}: {e}")
        return []

async def main():
    """Main function to benchmark both GPUs"""
    print("🚀 COMPREHENSIVE GPU COMPARISON BENCHMARK")
    print("="*80)
    
    # GPU configurations
    gpus_to_test = [
        ("http://192.168.2.180:1234", "RTX 3080"),
        ("http://192.168.2.64:1234", "RTX 4090")
    ]
    
    all_gpu_results = {}
    
    # Test each GPU
    for gpu_host, gpu_name in gpus_to_test:
        try:
            results = await run_gpu_benchmark(gpu_host, gpu_name)
            all_gpu_results[gpu_name] = results
        except Exception as e:
            logger.error(f"Failed to benchmark {gpu_name}: {e}")
            all_gpu_results[gpu_name] = []
    
    # Generate comparison summary
    print("\n" + "="*80)
    print("📈 GPU PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'GPU':<15} {'Users':<6} {'QPS':<8} {'Avg (ms)':<10} {'P95 (ms)':<10} {'Success':<8}")
    print("-" * 70)
    
    for gpu_name, results in all_gpu_results.items():
        for scenario in results:
            s = scenario['summary']
            print(f"{gpu_name:<15} {s.concurrent_users:<6} {s.queries_per_second:<8.1f} "
                  f"{s.avg_total_time*1000:<10.0f} {s.p95_time*1000:<10.0f} "
                  f"{s.successful_queries/s.total_queries*100:<7.0f}%")
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gpu_comparison_benchmark_{timestamp}.json"
    
    comprehensive_data = {
        'timestamp': datetime.now().isoformat(),
        'gpu_results': all_gpu_results,
        'system_info': {
            'tested_gpus': gpus_to_test,
            'vector_db': config.vector_db_path,
            'collection': config.collection_name
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(comprehensive_data, f, indent=2, default=str)
    
    print(f"\n💾 GPU comparison results saved to: {filename}")
    print("\n🎉 GPU benchmark comparison complete!")

if __name__ == "__main__":
    asyncio.run(main())