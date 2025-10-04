#!/usr/bin/env python3
"""
Batch Document Processor
Processes all PDF documents in the uploads directory that haven't been ingested yet.
"""

import os
import sys
import logging
from pathlib import Path
from hybrid_rag import HybridRAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    print("🔄 BATCH DOCUMENT PROCESSING")
    print("=" * 50)
    print()
    
    try:
        # Initialize the RAG system
        print("🔧 Initializing Enhanced RAG System...")
        rag = HybridRAGSystem()
        print("✅ System initialized successfully")
        print()
        
        # Check current state
        initial_count = rag.vector_db.collection.count()
        print(f"📊 Current documents in database: {initial_count}")
        
        # Check knowledge graph state
        try:
            kg_nodes = len(rag.knowledge_graph.graph.nodes())
            kg_edges = len(rag.knowledge_graph.graph.edges())
            print(f"🧠 Knowledge graph: {kg_nodes} nodes, {kg_edges} relationships")
        except:
            print("🧠 Knowledge graph: Empty/not loaded")
        print()
        
        # Get all PDF files in uploads
        uploads_dir = Path("uploads")
        if not uploads_dir.exists():
            print("❌ No uploads directory found!")
            return
            
        pdf_files = list(uploads_dir.glob("*.pdf"))
        if not pdf_files:
            print("📭 No PDF files found in uploads directory")
            return
            
        print(f"📄 Found {len(pdf_files)} PDF files to process:")
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"   {i:2d}. {pdf_file.name}")
        print()
        
        # Process each document
        processed_count = 0
        failed_count = 0
        
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"🔄 Processing {i}/{len(pdf_files)}: {pdf_file.name[:60]}...")
            
            try:
                # Process single document
                success = rag.ingest_single_document(str(pdf_file))
                
                if success:
                    processed_count += 1
                    print(f"   ✅ Successfully processed")
                else:
                    failed_count += 1
                    print(f"   ❌ Failed to process")
                    
            except Exception as e:
                failed_count += 1
                print(f"   ❌ Error: {str(e)[:100]}...")
                
            # Progress update
            if i % 5 == 0 or i == len(pdf_files):
                final_count = rag.vector_db.collection.count()
                added = final_count - initial_count
                print(f"   📊 Progress: {added} documents added to database")
                print()
        
        # Final summary
        print("=" * 50)
        print("🎉 BATCH PROCESSING COMPLETE!")
        print()
        
        final_count = rag.vector_db.collection.count()
        total_added = final_count - initial_count
        
        print(f"📊 Results Summary:")
        print(f"   • Files processed successfully: {processed_count}")
        print(f"   • Files failed: {failed_count}")
        print(f"   • Total documents in database: {final_count}")
        print(f"   • New documents added: {total_added}")
        
        # Check final knowledge graph state
        try:
            final_kg_nodes = len(rag.knowledge_graph.graph.nodes())
            final_kg_edges = len(rag.knowledge_graph.graph.edges())
            print(f"   • Knowledge graph: {final_kg_nodes} nodes, {final_kg_edges} relationships")
        except Exception as e:
            print(f"   • Knowledge graph: Error loading ({str(e)})")
            
        print()
        if processed_count > 0:
            print("✅ Your enhanced RAG system is now ready with semantic medical intelligence!")
            print("🎯 You can now query your medical documents with advanced understanding.")
        else:
            print("⚠️  No documents were successfully processed. Check the error messages above.")
            
    except Exception as e:
        print(f"❌ System error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()