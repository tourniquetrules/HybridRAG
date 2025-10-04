#!/usr/bin/env python3
"""
Document Processing Status Checker
Quick utility to check document processing status
"""

import requests
import json
from datetime import datetime

def get_processing_summary():
    """Get and display document processing summary"""
    try:
        # Check if server is running
        response = requests.get("http://localhost:5000/api/documents/summary", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            print("=" * 80)
            print("📊 DOCUMENT PROCESSING STATUS REPORT")
            print("=" * 80)
            print(f"🕐 Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # System Overview
            print("🔧 SYSTEM STATUS:")
            system_status = data.get('system_status', {})
            print(f"   • Vector Database: {system_status.get('vector_database', 'UNKNOWN')}")
            print(f"   • Knowledge Graph: {system_status.get('knowledge_graph', 'UNKNOWN')}")
            print(f"   • Document Processor: {system_status.get('document_processor', 'UNKNOWN')}")
            print()
            
            # Summary Stats
            print("📈 PROCESSING SUMMARY:")
            print(f"   • Total Documents Processed: {data.get('total_documents_processed', 0)}")
            print(f"   • Total Chunks Created: {data.get('total_chunks', 0)}")
            print(f"   • Knowledge Graph Nodes: {data.get('knowledge_graph_nodes', 0)}")
            print(f"   • Knowledge Graph Relationships: {data.get('knowledge_graph_relationships', 0)}")
            print()
            
            # Document Details
            documents = data.get('documents', [])
            if documents:
                print("📋 DOCUMENT DETAILS:")
                print("-" * 80)
                for i, doc in enumerate(documents, 1):
                    name = doc.get('document_name', 'Unknown')
                    chunks = doc.get('chunks_created', 0)
                    status = doc.get('status', 'UNKNOWN')
                    date = doc.get('processing_date', 'unknown')
                    
                    # Truncate long names
                    display_name = name if len(name) <= 50 else name[:47] + "..."
                    
                    status_emoji = "✅" if status == "SUCCESS" else "❌"
                    print(f"{i:2d}. {status_emoji} {display_name}")
                    print(f"     Chunks: {chunks} | Status: {status} | Date: {date}")
                    
                    chunk_types = doc.get('chunk_types', [])
                    if chunk_types:
                        print(f"     Chunk Types: {', '.join(chunk_types)}")
                    print()
            else:
                print("📋 DOCUMENT DETAILS:")
                print("   No documents processed yet.")
                print()
            
            print("=" * 80)
            
        else:
            print(f"❌ Error: Server returned status code {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to server at http://localhost:5000")
        print("Make sure the FastAPI server is running.")
    except requests.exceptions.Timeout:
        print("❌ Error: Request timed out. Server may be busy.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    get_processing_summary()