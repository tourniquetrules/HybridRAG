#!/usr/bin/env python3
"""
Benchmark Results Viewer and Analyzer
Easily view and compare saved benchmark results
"""

import json
import os
import pandas as pd
from datetime import datetime
import argparse

def list_benchmark_files():
    """List all available benchmark files"""
    benchmark_files = []
    for file in os.listdir('.'):
        if 'benchmark' in file and file.endswith('.json'):
            benchmark_files.append(file)
    
    benchmark_files.sort()
    return benchmark_files

def load_benchmark_data(filename):
    """Load benchmark data from JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def display_comprehensive_benchmark(data):
    """Display comprehensive benchmark results in a formatted way"""
    print("\n" + "="*80)
    print(f"📊 COMPREHENSIVE BENCHMARK ANALYSIS")
    print(f"📅 Date: {data['metadata']['date']}")
    print(f"📝 Description: {data['metadata']['description']}")
    print("="*80)
    
    print(f"\n🎯 KEY FINDINGS:")
    for finding in data['analysis']['key_findings']:
        print(f"   • {finding}")
    
    print(f"\n🏆 CHAMPIONS:")
    champions = data['analysis']['speed_champions']
    
    print(f"\n   🥇 Overall Speed Champion: {champions['overall_fastest']['config']}")
    print(f"      • Single User: {champions['overall_fastest']['single_user_time']}")
    print(f"      • 5 Users: {champions['overall_fastest']['concurrent_performance']}")
    print(f"      • QPS: {champions['overall_fastest']['qps']}")
    
    print(f"\n   🏥 Medical Champion: {champions['medical_champion']['config']}")
    print(f"      • Single User: {champions['medical_champion']['single_user_time']}")
    print(f"      • 5 Users: {champions['medical_champion']['concurrent_performance']}")
    print(f"      • QPS: {champions['medical_champion']['qps']}")
    
    print(f"\n📈 PERFORMANCE SUMMARY:")
    print(f"{'GPU + Model':<30} {'1 User':<10} {'3 Users':<10} {'5 Users':<10} {'Max QPS':<8} {'Specialty':<12}")
    print("-" * 85)
    
    # RTX 4090 Results
    for model_key, model_data in data['benchmark_results']['rtx_4090']['model_performance'].items():
        gpu_model = f"RTX 4090 + {model_data['model']}"
        perf = model_data['performance']
        max_qps = max(perf['1_user']['qps'], perf['3_users']['qps'], perf['5_users']['qps'])
        
        print(f"{gpu_model:<30} {perf['1_user']['llm_time_ms']/1000:.2f}s    "
              f"{perf['3_users']['llm_time_ms']/1000:.2f}s    "
              f"{perf['5_users']['llm_time_ms']/1000:.2f}s    "
              f"{max_qps:<8.1f} {model_data['specialization']:<12}")
    
    # RTX 3080 Results
    for model_key, model_data in data['benchmark_results']['rtx_3080']['model_performance'].items():
        gpu_model = f"RTX 3080 + {model_data['model']}"
        perf = model_data['performance']
        max_qps = max(perf['1_user']['qps'], perf['3_users']['qps'], perf['5_users']['qps'])
        
        champion_mark = " 👑" if model_key == 'lfm_1_2b' else ""
        
        print(f"{gpu_model + champion_mark:<30} {perf['1_user']['llm_time_ms']/1000:.2f}s    "
              f"{perf['3_users']['llm_time_ms']/1000:.2f}s    "
              f"{perf['5_users']['llm_time_ms']/1000:.2f}s    "
              f"{max_qps:<8.1f} {model_data['specialization']:<12}")
    
    print(f"\n💡 PRODUCTION RECOMMENDATIONS:")
    recs = data['analysis']['production_recommendations']
    print(f"   🚀 High Traffic General: {recs['high_traffic_general']}")
    print(f"   🏥 Medical Professional: {recs['medical_professional']}")
    print(f"   💰 Budget Medical: {recs['budget_medical']}")
    print(f"   ❌ Avoid: {recs['avoid']}")

def display_individual_benchmark(data):
    """Display individual benchmark results"""
    print(f"\n📊 INDIVIDUAL BENCHMARK: {data['gpu_name']}")
    print(f"🖥️  Host: {data['gpu_host']}")
    print(f"📅 Timestamp: {data['timestamp']}")
    print("-" * 60)
    
    for scenario in data['scenarios']:
        s = scenario['summary']
        print(f"\n👥 {s.concurrent_users} Concurrent Users:")
        print(f"   ✅ Success: {s.successful_queries}/{s.total_queries} ({s.successful_queries/s.total_queries*100:.1f}%)")
        print(f"   ⚡ QPS: {s.queries_per_second:.1f}")
        print(f"   ⏱️  Average Response: {s.avg_total_time*1000:.0f}ms")
        if hasattr(s, 'avg_llm_inference_time') and s.avg_llm_inference_time:
            print(f"   🧠 LLM Time: {s.avg_llm_inference_time*1000:.0f}ms")
            print(f"   🔍 Search Time: {s.avg_search_time*1000:.0f}ms")

def main():
    parser = argparse.ArgumentParser(description='View and analyze benchmark results')
    parser.add_argument('--list', action='store_true', help='List all available benchmark files')
    parser.add_argument('--file', type=str, help='Specific benchmark file to analyze')
    parser.add_argument('--latest', action='store_true', help='Show the latest comprehensive benchmark')
    parser.add_argument('--compare', action='store_true', help='Compare all benchmark files')
    
    args = parser.parse_args()
    
    benchmark_files = list_benchmark_files()
    
    if args.list:
        print("\n📁 Available Benchmark Files:")
        for i, file in enumerate(benchmark_files, 1):
            file_size = os.path.getsize(file) / 1024
            mod_time = datetime.fromtimestamp(os.path.getmtime(file))
            print(f"   {i}. {file} ({file_size:.1f}KB) - {mod_time.strftime('%Y-%m-%d %H:%M')}")
        return
    
    if args.file:
        if os.path.exists(args.file):
            data = load_benchmark_data(args.file)
            if data:
                if 'metadata' in data:
                    display_comprehensive_benchmark(data)
                else:
                    display_individual_benchmark(data)
        else:
            print(f"❌ File not found: {args.file}")
        return
    
    if args.latest:
        # Find the latest comprehensive benchmark
        comprehensive_files = [f for f in benchmark_files if 'comprehensive' in f]
        if comprehensive_files:
            latest_file = max(comprehensive_files, key=lambda f: os.path.getmtime(f))
            data = load_benchmark_data(latest_file)
            if data:
                display_comprehensive_benchmark(data)
        else:
            print("❌ No comprehensive benchmark files found")
        return
    
    if args.compare:
        print("\n📊 BENCHMARK COMPARISON SUMMARY")
        print("="*80)
        # This would implement comparison logic between multiple files
        print("Feature coming soon...")
        return
    
    # Default: show help and list files
    parser.print_help()
    print("\n📁 Available Files:")
    for i, file in enumerate(benchmark_files, 1):
        file_size = os.path.getsize(file) / 1024
        print(f"   {i}. {file} ({file_size:.1f}KB)")

if __name__ == "__main__":
    main()