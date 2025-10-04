#!/usr/bin/env python3
"""
Database & System Reset Script
Clears all databases, knowledge graphs, uploads, and logs for fresh testing
"""

import os
import shutil
import sys
from pathlib import Path
import requests
import time
import subprocess

def check_server_status():
    """Check if the server is running"""
    try:
        response = requests.get("http://localhost:5000/api/status", timeout=2)
        return response.status_code == 200
    except:
        return False

def stop_server():
    """Stop the FastAPI server"""
    print("🔄 Stopping FastAPI server...")
    try:
        result = subprocess.run(["pkill", "-f", "fastapi_app.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Server stopped")
            time.sleep(2)  # Give it time to shutdown cleanly
        else:
            print("ℹ️  No server process found (may already be stopped)")
    except Exception as e:
        print(f"⚠️  Error stopping server: {e}")

def clear_vector_database():
    """Clear ChromaDB vector database"""
    print("🗑️  Clearing vector database...")
    try:
        # Remove ChromaDB files
        vector_db_path = Path("vector_db")
        
        if vector_db_path.exists():
            # Remove specific ChromaDB files
            chroma_files = [
                vector_db_path / "chroma.sqlite3",
                vector_db_path / "chroma.sqlite3-shm",
                vector_db_path / "chroma.sqlite3-wal"
            ]
            
            for file in chroma_files:
                if file.exists():
                    file.unlink()
                    print(f"   ✅ Removed {file.name}")
            
            # Remove any ChromaDB directories
            for item in vector_db_path.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                    print(f"   ✅ Removed directory {item.name}")
        else:
            print("   ℹ️  Vector DB directory doesn't exist")
            
        print("✅ Vector database cleared")
    except Exception as e:
        print(f"❌ Error clearing vector database: {e}")

def clear_knowledge_graph():
    """Clear NetworkX knowledge graph"""
    print("🧠 Clearing knowledge graph...")
    try:
        kg_file = Path("vector_db") / "knowledge_graph.pkl"
        if kg_file.exists():
            kg_file.unlink()
            print("   ✅ Removed knowledge_graph.pkl")
            print("✅ Knowledge graph cleared")
        else:
            print("   ℹ️  Knowledge graph file doesn't exist")
    except Exception as e:
        print(f"❌ Error clearing knowledge graph: {e}")

def clear_uploads():
    """Clear uploads directory"""
    print("📁 Clearing uploads directory...")
    try:
        uploads_path = Path("uploads")
        if uploads_path.exists():
            file_count = 0
            for file in uploads_path.iterdir():
                if file.is_file():
                    file.unlink()
                    file_count += 1
            print(f"   ✅ Removed {file_count} uploaded files")
            print("✅ Uploads directory cleared")
        else:
            print("   ℹ️  Uploads directory doesn't exist")
    except Exception as e:
        print(f"❌ Error clearing uploads: {e}")

def clear_logs():
    """Clear log files"""
    print("📄 Clearing log files...")
    try:
        # Clear main server log
        server_log = Path("server.log")
        if server_log.exists():
            server_log.unlink()
            print("   ✅ Removed server.log")
        
        # Clear document processing log
        doc_log = Path("logs") / "document_processing.log"
        if doc_log.exists():
            doc_log.unlink()
            print("   ✅ Removed document_processing.log")
        
        # Clear any other logs in logs directory
        logs_dir = Path("logs")
        if logs_dir.exists():
            log_count = 0
            for log_file in logs_dir.iterdir():
                if log_file.is_file():
                    log_file.unlink()
                    log_count += 1
            if log_count > 0:
                print(f"   ✅ Removed {log_count} additional log files")
        
        print("✅ Log files cleared")
    except Exception as e:
        print(f"❌ Error clearing logs: {e}")

def clear_cache():
    """Clear Python cache files"""
    print("🧹 Clearing Python cache...")
    try:
        cache_count = 0
        # Remove __pycache__ directories
        for pycache_dir in Path(".").rglob("__pycache__"):
            if pycache_dir.is_dir():
                shutil.rmtree(pycache_dir)
                cache_count += 1
        
        # Remove .pyc files
        for pyc_file in Path(".").rglob("*.pyc"):
            pyc_file.unlink()
            cache_count += 1
        
        if cache_count > 0:
            print(f"   ✅ Removed {cache_count} cache files/directories")
        else:
            print("   ℹ️  No cache files found")
        print("✅ Python cache cleared")
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting FastAPI server...")
    try:
        # Use bash to properly activate the virtual environment
        cmd = [
            "/bin/bash", "-c",
            "source venv_py310/bin/activate && exec python fastapi_app.py"
        ]
        
        # Start server in background with nohup
        with open("server.log", "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # Detach from parent process
                cwd=os.getcwd()  # Make sure we're in the right directory
            )
        
        print("✅ Server starting...")
        
        # Wait for server to be ready
        print("⏳ Waiting for server to initialize...")
        for i in range(15):
            if check_server_status():
                print("✅ Server is ready!")
                return True
            time.sleep(1)
            print(f"   Waiting... ({i+1}/15)")
        
        print("⚠️  Server may still be starting up. Check with: curl http://localhost:5000/api/status")
        return False
        
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

def main():
    """Main reset function"""
    # Check for help
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
        print("🔄 Database & System Reset Script")
        print()
        print("Usage: python reset_system.py [OPTIONS]")
        print()
        print("Options:")
        print("  --force, -f       Skip confirmation prompt")
        print("  --no-restart     Don't restart server after reset")
        print("  --auto           Force reset and auto-restart (non-interactive)")
        print("  --help, -h       Show this help")
        print()
        print("Examples:")
        print("  python reset_system.py                    # Interactive reset")
        print("  python reset_system.py --auto            # Full auto reset")
        print("  python reset_system.py --force           # Skip confirmation only")
        print("  python reset_system.py --no-restart     # Reset but don't restart server")
        return
    
    print("=" * 80)
    print("🔄 DATABASE & SYSTEM RESET SCRIPT")
    print("=" * 80)
    print("This will clear:")
    print("  • Vector Database (ChromaDB)")
    print("  • Knowledge Graph (NetworkX)")
    print("  • Uploaded Files")
    print("  • Log Files")
    print("  • Python Cache")
    print("=" * 80)
    
    # Check for auto mode
    auto_mode = "--auto" in sys.argv
    force_mode = "--force" in sys.argv or "-f" in sys.argv or auto_mode
    
    # Confirm action
    if force_mode:
        if auto_mode:
            print("🤖 Auto mode: Proceeding with full reset and restart...")
        else:
            print("🔥 Force mode: Proceeding without confirmation...")
    else:
        try:
            confirm = input("Are you sure you want to clear everything? (y/N): ")
            if confirm.lower() not in ['y', 'yes']:
                print("❌ Reset cancelled")
                return
        except EOFError:
            print("❌ Reset cancelled (no input)")
            return
    
    print("\n🔄 Starting system reset...")
    
    # Check if server is running
    if check_server_status():
        print("⚠️  Server is running")
        stop_server()
    else:
        print("ℹ️  Server is not running")
    
    # Clear everything
    clear_vector_database()
    clear_knowledge_graph()
    clear_uploads()
    clear_logs()
    clear_cache()
    
    print("\n🎯 Reset complete!")
    
    # Ask if user wants to restart server
    no_restart = "--no-restart" in sys.argv
    auto_mode = "--auto" in sys.argv
    
    if no_restart:
        print("🔄 Server restart skipped (--no-restart flag)")
        should_restart = False
    elif auto_mode:
        print("🤖 Auto mode: Restarting server automatically...")
        should_restart = True
    else:
        try:
            restart = input("\nRestart server now? (Y/n): ")
            should_restart = restart.lower() not in ['n', 'no']
        except EOFError:
            # Default to yes if no input (e.g., from piped input)
            print("Y")  # Show what was chosen
            should_restart = True
    
    if should_restart:
        start_server()
        print("\n📊 Check system status with: python check_documents.py")
    else:
        print("🔄 Server not restarted. Start manually when ready:")
        print("   ./venv_py310/bin/python fastapi_app.py")
    
    print("\n" + "=" * 80)
    print("✅ System is ready for fresh testing!")
    print("🌐 Web interface: http://localhost:5000")
    print("📊 Check status: python check_documents.py")
    print("📄 View logs: python view_document_log.py")
    print("=" * 80)

if __name__ == "__main__":
    main()