#!/usr/bin/env python3
"""
System Status Overview
Shows current state of databases, files, and logs
"""

import os
from pathlib import Path
import requests

def check_vector_database():
    """Check vector database status"""
    print("💾 VECTOR DATABASE:")
    vector_db_path = Path("vector_db")
    
    if not vector_db_path.exists():
        print("   ❌ No vector_db directory")
        return
    
    # Check for ChromaDB files
    chroma_files = [
        vector_db_path / "chroma.sqlite3",
        vector_db_path / "chroma.sqlite3-shm", 
        vector_db_path / "chroma.sqlite3-wal"
    ]
    
    found_files = [f for f in chroma_files if f.exists()]
    if found_files:
        print(f"   ✅ ChromaDB files: {len(found_files)} found")
        for f in found_files:
            size = f.stat().st_size if f.exists() else 0
            print(f"      • {f.name}: {size:,} bytes")
    else:
        print("   📭 No ChromaDB files found")
    
    # Check for directories
    dirs = [d for d in vector_db_path.iterdir() if d.is_dir()]
    if dirs:
        print(f"   📁 Directories: {len(dirs)}")
        for d in dirs:
            print(f"      • {d.name}")

def check_knowledge_graph():
    """Check knowledge graph status"""
    print("\n🧠 KNOWLEDGE GRAPH:")
    kg_file = Path("vector_db") / "knowledge_graph.pkl"
    
    if kg_file.exists():
        size = kg_file.stat().st_size
        print(f"   ✅ NetworkX graph file: {size:,} bytes")
    else:
        print("   📭 No knowledge graph file found")

def check_uploads():
    """Check uploads directory"""
    print("\n📁 UPLOADS DIRECTORY:")
    uploads_path = Path("uploads")
    
    if not uploads_path.exists():
        print("   ❌ No uploads directory")
        return
    
    files = list(uploads_path.glob("*.pdf"))
    if files:
        print(f"   ✅ PDF files: {len(files)}")
        total_size = sum(f.stat().st_size for f in files)
        print(f"   📊 Total size: {total_size:,} bytes")
        
        # Show recent files
        recent_files = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]
        print("   📋 Recent files:")
        for f in recent_files:
            size = f.stat().st_size
            name = f.name if len(f.name) <= 50 else f.name[:47] + "..."
            print(f"      • {name} ({size:,} bytes)")
    else:
        print("   📭 No PDF files found")

def check_logs():
    """Check log files"""
    print("\n📄 LOG FILES:")
    
    # Main server log
    server_log = Path("server.log")
    if server_log.exists():
        size = server_log.stat().st_size
        print(f"   ✅ server.log: {size:,} bytes")
    else:
        print("   📭 No server.log found")
    
    # Document processing log
    doc_log = Path("logs") / "document_processing.log"
    if doc_log.exists():
        size = doc_log.stat().st_size
        print(f"   ✅ document_processing.log: {size:,} bytes")
        
        # Count entries
        try:
            with open(doc_log, 'r') as f:
                lines = [l for l in f.readlines() if l.strip() and not l.startswith('=')]
                entries = len([l for l in lines if '|' in l and ('✅' in l or '❌' in l)])
            print(f"   📊 Document entries: {entries}")
        except:
            pass
    else:
        print("   📭 No document_processing.log found")
    
    # Other logs
    logs_dir = Path("logs")
    if logs_dir.exists():
        other_logs = [f for f in logs_dir.iterdir() 
                     if f.is_file() and f.name != "document_processing.log"]
        if other_logs:
            print(f"   📁 Other logs: {len(other_logs)}")

def check_server_status():
    """Check if server is running and get stats"""
    print("\n🌐 SERVER STATUS:")
    try:
        response = requests.get("http://localhost:5000/api/status", timeout=3)
        if response.status_code == 200:
            data = response.json()
            print("   ✅ Server is running")
            print(f"   📊 Vector DB documents: {data.get('vector_database', {}).get('total_documents', 0)}")
            kg_info = data.get('knowledge_graph', {})
            print(f"   🧠 Knowledge graph: {kg_info.get('nodes', 0)} nodes, {kg_info.get('relationships', 0)} relationships")
        else:
            print(f"   ⚠️  Server responding but status: {response.status_code}")
    except:
        print("   ❌ Server is not running")

def main():
    """Main status function"""
    print("=" * 80)
    print("📊 SYSTEM STATUS OVERVIEW")
    print("=" * 80)
    
    check_vector_database()
    check_knowledge_graph()
    check_uploads()
    check_logs()
    check_server_status()
    
    print("\n" + "=" * 80)
    print("🛠️  AVAILABLE COMMANDS:")
    print("   🔄 Reset everything: python reset_system.py")
    print("   📄 View document log: python view_document_log.py")
    print("   📊 Check documents: python check_documents.py")
    print("   🌐 Web interface: http://localhost:5000")
    print("=" * 80)

if __name__ == "__main__":
    main()