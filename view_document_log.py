#!/usr/bin/env python3
"""
Document Processing Log Viewer
Simple utility to view the dedicated document processing log
"""

import os
from pathlib import Path

def view_document_log():
    """View the document processing log"""
    log_file = Path("logs") / "document_processing.log"
    
    if not log_file.exists():
        print("📄 No document processing log found yet.")
        print("Documents will be logged here when you upload them.")
        print(f"Expected location: {log_file}")
        return
    
    print("=" * 80)
    print("📄 DOCUMENT PROCESSING LOG")
    print("=" * 80)
    
    try:
        with open(log_file, 'r') as f:
            content = f.read().strip()
            if content:
                print(content)
            else:
                print("Log file exists but is empty.")
                
    except Exception as e:
        print(f"❌ Error reading log file: {str(e)}")
    
    print("=" * 80)

def tail_document_log():
    """Tail the document processing log (last 20 lines)"""
    log_file = Path("logs") / "document_processing.log"
    
    if not log_file.exists():
        print("📄 No document processing log found yet.")
        return
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            
        # Show last 20 lines
        recent_lines = lines[-20:] if len(lines) > 20 else lines
        
        print("=" * 80)
        print("📄 RECENT DOCUMENT PROCESSING (Last 20 entries)")
        print("=" * 80)
        
        for line in recent_lines:
            print(line.rstrip())
            
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error reading log file: {str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--tail":
        tail_document_log()
    else:
        view_document_log()