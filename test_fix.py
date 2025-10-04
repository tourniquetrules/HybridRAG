#!/usr/bin/env python3
"""
Test script to verify the method name fix is working
"""

from hybrid_rag import HybridRAGSystem
import tempfile
import os

def create_test_pdf():
    """Create a simple test PDF using reportlab"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            c = canvas.Canvas(tmp.name, pagesize=letter)
            c.drawString(100, 750, "Test Document for Method Fix Verification")
            c.drawString(100, 700, "")
            c.drawString(100, 650, "This is a test document to verify that the EnhancedDocumentProcessor")
            c.drawString(100, 620, "extract_chunks() method is working correctly after fixing the")
            c.drawString(100, 590, "method name mismatch issue.")
            c.drawString(100, 560, "")
            c.drawString(100, 530, "Medical Terms Test:")
            c.drawString(100, 500, "- Sepsis management")
            c.drawString(100, 470, "- Emergency medicine protocols")
            c.drawString(100, 440, "- Patient assessment procedures")
            c.drawString(100, 410, "- Clinical decision making")
            c.save()
            
        return tmp.name
        
    except ImportError:
        # Fallback: create a simple text file if reportlab not available
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            tmp.write("""Test Document for Method Fix Verification

This is a test document to verify that the EnhancedDocumentProcessor
extract_chunks() method is working correctly after fixing the
method name mismatch issue.

Medical Terms Test:
- Sepsis management
- Emergency medicine protocols  
- Patient assessment procedures
- Clinical decision making

The semantic chunking should now work properly with spaCy SciBERT
for medical entity recognition and enhanced document processing.
""")
        return tmp.name

def main():
    print("🔧 TESTING METHOD NAME FIX")
    print("=" * 40)
    print()
    
    try:
        # Initialize the RAG system
        print("🔄 Initializing HybridRAG system...")
        rag = HybridRAGSystem()
        print("✅ System initialized successfully")
        
        # Verify processor type and methods
        processor = rag.document_processor
        print(f"✅ Processor type: {type(processor).__name__}")
        
        # Check for correct method
        if hasattr(processor, 'extract_chunks'):
            print("✅ extract_chunks() method found")
        else:
            print("❌ extract_chunks() method missing")
            
        if hasattr(processor, 'extract_text_chunks'):
            print("⚠️  Old extract_text_chunks() method still present")
        else:
            print("✅ Old extract_text_chunks() method properly removed/renamed")
        
        # Create test document
        print("\n🔄 Creating test document...")
        test_file = create_test_pdf()
        print(f"✅ Test document created: {test_file}")
        
        # Test document ingestion
        print("\n🔄 Testing document ingestion...")
        try:
            result = rag.ingest_single_document(test_file)
            if result:
                print("✅ SUCCESS: Document ingested successfully!")
                
                # Check database
                doc_count = rag.vector_db.collection.count()
                print(f"📊 Total documents in database: {doc_count}")
                
                # Test a simple query
                print("\n🔄 Testing query functionality...")
                query_result = rag.search("sepsis management", max_results=3)
                if query_result and len(query_result) > 0:
                    print(f"✅ Query successful: Found {len(query_result)} results")
                    for i, result in enumerate(query_result[:2]):
                        # Handle result display based on its type
                        if isinstance(result, dict):
                            text = result.get('text', str(result))[:100] + "..."
                            print(f"   Result {i+1}: {text}")
                        else:
                            print(f"   Result {i+1}: {str(result)[:100]}...")
                else:
                    print("⚠️  Query returned no results (database might be empty)")
                    
            else:
                print("❌ Document ingestion returned False")
                
        except AttributeError as e:
            if "extract_text_chunks" in str(e):
                print(f"❌ METHOD NAME ERROR STILL EXISTS: {e}")
            else:
                print(f"❌ Attribute error: {e}")
        except Exception as e:
            print(f"❌ Ingestion error: {e}")
        
        # Cleanup
        if os.path.exists(test_file):
            os.unlink(test_file)
            print(f"🧹 Cleaned up test file: {test_file}")
            
    except Exception as e:
        print(f"❌ System initialization error: {e}")
    
    print("\n" + "=" * 40)
    print("🏁 Test complete!")

if __name__ == "__main__":
    main()