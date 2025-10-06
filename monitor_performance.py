#!/usr/bin/env python3
"""
Real-time Vector DB Performance Monitor
Monitors vector database performance in real-time during operation
"""

import time
import threading
import psutil
import json
from datetime import datetime
from collections import deque
import signal
import sys

from hybrid_rag import HybridRAGSystem

class PerformanceMonitor:
    def __init__(self, max_history=100):
        self.rag_system = HybridRAGSystem()
        self.max_history = max_history
        self.query_times = deque(maxlen=max_history)
        self.system_metrics = deque(maxlen=max_history)
        self.running = False
        
    def monitor_query(self, query: str) -> dict:
        """Monitor a single query and record metrics"""
        start_time = time.time()
        
        try:
            # Time the embedding
            embed_start = time.time()
            embedding = self.rag_system.vector_db.embedding_model.encode([query]).tolist()
            embed_time = time.time() - embed_start
            
            # Time the search
            search_start = time.time()
            results = self.rag_system.vector_db.collection.query(
                query_embeddings=[embedding],
                n_results=5,
                include=['documents', 'metadatas', 'distances']
            )
            search_time = time.time() - search_start
            
            total_time = time.time() - start_time
            
            # Record metrics
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'query': query[:50] + "..." if len(query) > 50 else query,
                'embedding_time': embed_time,
                'search_time': search_time,
                'total_time': total_time,
                'results_count': len(results['documents'][0]) if results['documents'] else 0,
                'success': True
            }
            
            self.query_times.append(metrics)
            return metrics
            
        except Exception as e:
            error_metrics = {
                'timestamp': datetime.now().isoformat(),
                'query': query[:50] + "..." if len(query) > 50 else query,
                'embedding_time': 0,
                'search_time': 0,
                'total_time': time.time() - start_time,
                'results_count': 0,
                'success': False,
                'error': str(e)
            }
            self.query_times.append(error_metrics)
            return error_metrics
    
    def collect_system_metrics(self):
        """Collect current system metrics"""
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'disk_io_read': psutil.disk_io_counters().read_bytes if psutil.disk_io_counters() else 0,
            'disk_io_write': psutil.disk_io_counters().write_bytes if psutil.disk_io_counters() else 0,
        }
    
    def start_monitoring(self, interval=1.0):
        """Start background system monitoring"""
        self.running = True
        
        def monitor_loop():
            while self.running:
                metrics = self.collect_system_metrics()
                self.system_metrics.append(metrics)
                time.sleep(interval)
        
        monitor_thread = threading.Thread(target=monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.running = False
    
    def get_current_stats(self) -> dict:
        """Get current performance statistics"""
        if not self.query_times:
            return {
                'query_count': 0,
                'avg_total_time': 0,
                'avg_embedding_time': 0,
                'avg_search_time': 0,
                'success_rate': 0,
                'queries_per_minute': 0
            }
        
        recent_queries = list(self.query_times)[-20:]  # Last 20 queries
        successful_queries = [q for q in recent_queries if q['success']]
        
        if not successful_queries:
            return {
                'query_count': len(recent_queries),
                'avg_total_time': 0,
                'avg_embedding_time': 0,
                'avg_search_time': 0,
                'success_rate': 0,
                'queries_per_minute': 0
            }
        
        # Calculate time-based metrics
        now = datetime.now()
        one_minute_ago = now.timestamp() - 60
        recent_minute_queries = [
            q for q in recent_queries 
            if datetime.fromisoformat(q['timestamp']).timestamp() > one_minute_ago
        ]
        
        return {
            'query_count': len(self.query_times),
            'avg_total_time': sum(q['total_time'] for q in successful_queries) / len(successful_queries),
            'avg_embedding_time': sum(q['embedding_time'] for q in successful_queries) / len(successful_queries),
            'avg_search_time': sum(q['search_time'] for q in successful_queries) / len(successful_queries),
            'success_rate': len(successful_queries) / len(recent_queries),
            'queries_per_minute': len(recent_minute_queries)
        }
    
    def print_dashboard(self):
        """Print real-time performance dashboard"""
        stats = self.get_current_stats()
        current_sys = self.system_metrics[-1] if self.system_metrics else {}
        
        # Clear screen
        print("\033[2J\033[H", end="")
        
        print("🔍 VECTOR DATABASE PERFORMANCE MONITOR")
        print("=" * 60)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        print("📊 QUERY PERFORMANCE:")
        print(f"  Total Queries:        {stats['query_count']}")
        print(f"  Success Rate:         {stats['success_rate']*100:.1f}%")
        print(f"  Queries/Minute:       {stats['queries_per_minute']}")
        print(f"  Avg Total Time:       {stats['avg_total_time']*1000:.1f}ms")
        print(f"  Avg Embedding Time:   {stats['avg_embedding_time']*1000:.1f}ms")
        print(f"  Avg Search Time:      {stats['avg_search_time']*1000:.1f}ms")
        print()
        
        print("🖥️  SYSTEM RESOURCES:")
        if current_sys:
            print(f"  CPU Usage:            {current_sys.get('cpu_percent', 0):.1f}%")
            print(f"  Memory Usage:         {current_sys.get('memory_percent', 0):.1f}%")
            print(f"  Memory Used:          {current_sys.get('memory_used_gb', 0):.2f} GB")
        print()
        
        print("📈 RECENT QUERIES:")
        recent_queries = list(self.query_times)[-5:]
        for q in reversed(recent_queries):
            status = "✅" if q['success'] else "❌"
            timestamp = datetime.fromisoformat(q['timestamp']).strftime('%H:%M:%S')
            print(f"  {status} {timestamp} | {q['total_time']*1000:6.1f}ms | {q['query']}")
        
        print()
        print("Press Ctrl+C to stop monitoring...")
    
    def save_report(self, filename=None):
        """Save monitoring data to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"monitoring_report_{timestamp}.json"
        
        report = {
            'summary': self.get_current_stats(),
            'query_history': list(self.query_times),
            'system_metrics': list(self.system_metrics),
            'generated_at': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Report saved to: {filename}")

def main():
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\n🛑 Stopping monitor...")
        monitor.stop_monitoring()
        monitor.save_report()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🚀 Starting Vector DB Performance Monitor")
    print("This will monitor queries made to the vector database.")
    print("Run queries through your application to see metrics.")
    print("\nPress Ctrl+C to stop and save report.\n")
    
    # Example queries for demonstration
    test_queries = [
        "What is sepsis?",
        "How do you treat anaphylaxis?",
        "Explain the Glasgow Coma Scale",
        "What are the signs of stroke?",
        "How do you manage cardiac arrest?"
    ]
    
    try:
        while True:
            monitor.print_dashboard()
            
            # Optionally run test queries for demonstration
            if len(monitor.query_times) < 5:
                import random
                test_query = random.choice(test_queries)
                monitor.monitor_query(test_query)
            
            time.sleep(2)  # Update every 2 seconds
            
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    main()