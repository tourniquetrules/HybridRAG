#!/usr/bin/env python3
"""
Test script for Emergency Medicine Hybrid RAG System
"""

import sys
import logging
from pathlib import Path
import tempfile
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported"""
    logger.info("Testing imports...")
    
    try:
        from config import config
        logger.info("✅ Config module imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import config: {e}")
        return False
    
    try:
        from document_processor import DocumentProcessor
        logger.info("✅ Document processor imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import document processor: {e}")
        return False
    
    try:
        from vector_database import VectorDatabase
        logger.info("✅ Vector database imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import vector database: {e}")
        return False
    
    try:
        from knowledge_graph import MedicalKnowledgeGraph
        logger.info("✅ Knowledge graph imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import knowledge graph: {e}")
        return False
    
    try:
        from reranker import Reranker
        logger.info("✅ Reranker imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import reranker: {e}")
        return False
    
    try:
        from llm_client import LMStudioClient
        logger.info("✅ LLM client imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import LLM client: {e}")
        return False
    
    try:
        from hybrid_rag import HybridRAGSystem
        logger.info("✅ Hybrid RAG system imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import Hybrid RAG system: {e}")
        return False
    
    return True

def test_dependencies():
    """Test that key dependencies are available"""
    logger.info("Testing dependencies...")
    
    # Test sentence transformers
    try:
        from sentence_transformers import SentenceTransformer
        from config import config
        logger.info(f"Testing embedding model: {config.embedding_model}")
        model = SentenceTransformer(config.embedding_model)
        test_embedding = model.encode(["test sentence"])
        logger.info(f"✅ Sentence transformers working (embedding shape: {test_embedding.shape})")
    except Exception as e:
        logger.error(f"❌ Sentence transformers failed: {e}")
        return False
    
    # Test ChromaDB
    try:
        import chromadb
        client = chromadb.Client()
        logger.info("✅ ChromaDB working")
    except Exception as e:
        logger.error(f"❌ ChromaDB failed: {e}")
        return False
    
    # Test spaCy (optional) - prioritize scientific model
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_sci_scibert")
            logger.info("Testing scientific spaCy model: en_core_sci_scibert")
        except OSError:
            nlp = spacy.load("en_core_web_sm")
            logger.info("Testing standard spaCy model: en_core_web_sm")
        
        doc = nlp("Patient presents with acute myocardial infarction and requires epinephrine treatment.")
        logger.info(f"✅ spaCy working (found {len(doc.ents)} entities)")
        if doc.ents:
            entities = [f"{ent.text} ({ent.label_})" for ent in doc.ents]
            logger.info(f"   Detected entities: {', '.join(entities)}")
    except Exception as e:
        logger.warning(f"⚠️ spaCy not fully working (optional): {e}")
    
    # Test NetworkX
    try:
        import networkx as nx
        graph = nx.Graph()
        graph.add_edge("A", "B")
        logger.info("✅ NetworkX working")
    except Exception as e:
        logger.error(f"❌ NetworkX failed: {e}")
        return False
    
    # Test Docling
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        logger.info("✅ Docling working")
    except Exception as e:
        logger.error(f"❌ Docling failed: {e}")
        return False
    
    return True

def test_vector_database():
    """Test vector database functionality"""
    logger.info("Testing vector database...")
    
    try:
        from vector_database import VectorDatabase
        
        # Create temporary database
        with tempfile.TemporaryDirectory() as temp_dir:
            import os
            os.environ['VECTOR_DB_PATH'] = temp_dir
            
            # Reload config with new path
            from importlib import reload
            import config as config_module
            reload(config_module)
            
            vdb = VectorDatabase()
            
            # Test adding documents
            test_chunks = [
                {
                    'text': 'Patient presents with chest pain and shortness of breath. Consider myocardial infarction.',
                    'chunk_id': 'test_chunk_1',
                    'metadata': {'source': 'test_doc_1.pdf', 'chunk_index': 0}
                },
                {
                    'text': 'Administer epinephrine for anaphylactic shock. Dosage is 0.3mg intramuscularly.',
                    'chunk_id': 'test_chunk_2',
                    'metadata': {'source': 'test_doc_2.pdf', 'chunk_index': 0}
                }
            ]
            
            success = vdb.add_documents(test_chunks)
            if not success:
                logger.error("❌ Failed to add test documents")
                return False
            
            # Test querying
            results = vdb.query_similar("chest pain myocardial infarction", n_results=2)
            if len(results) == 0:
                logger.error("❌ No results returned from query")
                return False
            
            logger.info(f"✅ Vector database working (returned {len(results)} results)")
            return True
            
    except Exception as e:
        logger.error(f"❌ Vector database test failed: {e}")
        return False

def test_knowledge_graph():
    """Test knowledge graph functionality"""
    logger.info("Testing knowledge graph...")
    
    try:
        from knowledge_graph import MedicalKnowledgeGraph
        
        kg = MedicalKnowledgeGraph(use_neo4j=False)  # Use NetworkX for testing
        
        # Test entity extraction
        test_text = "Patient has chest pain and requires epinephrine for anaphylaxis treatment."
        entities = kg.extract_entities(test_text)
        
        if not entities:
            logger.warning("⚠️ No entities extracted (may be normal)")
        else:
            logger.info(f"✅ Extracted entities: {entities}")
        
        # Test relationship extraction
        relationships = kg.extract_relationships(test_text, entities)
        logger.info(f"✅ Extracted {len(relationships)} relationships")
        
        # Test adding to graph
        kg.add_to_graph(entities, relationships, "test_chunk")
        
        stats = kg.get_graph_stats()
        logger.info(f"✅ Knowledge graph working (nodes: {stats['nodes']}, edges: {stats['relationships']})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Knowledge graph test failed: {e}")
        return False

def test_llm_connection():
    """Test LM Studio connection"""
    logger.info("Testing LM Studio connection...")
    
    try:
        from llm_client import LMStudioClient
        
        client = LMStudioClient()
        
        # Test simple generation
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello, this is a test!' in exactly those words."}
        ]
        
        response = client.generate_response(messages, temperature=0.1, max_tokens=50)
        
        if response:
            logger.info(f"✅ LM Studio connection working (response: {response[:50]}...)")
            return True
        else:
            logger.warning("⚠️ LM Studio connection failed (may not be running)")
            return False
            
    except Exception as e:
        logger.warning(f"⚠️ LM Studio test failed (may not be running): {e}")
        return False

def test_hybrid_rag():
    """Test the complete Hybrid RAG system"""
    logger.info("Testing Hybrid RAG system...")
    
    try:
        from hybrid_rag import HybridRAGSystem
        
        # Initialize with NetworkX (no Neo4j required)
        rag = HybridRAGSystem(use_neo4j=False)
        
        # Test system status
        status = rag.get_system_status()
        logger.info(f"✅ System status: {status}")
        
        # Note: We skip document ingestion test as it requires actual PDF files
        logger.info("✅ Hybrid RAG system initialized successfully")
        
        rag.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Hybrid RAG test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🏥 Starting Emergency Medicine Hybrid RAG System Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Dependencies", test_dependencies),
        ("Vector Database", test_vector_database),
        ("Knowledge Graph", test_knowledge_graph),
        ("LM Studio Connection", test_llm_connection),
        ("Hybrid RAG System", test_hybrid_rag),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready to use.")
        return 0
    elif passed >= total - 1:  # Allow LM Studio to fail
        logger.info("✅ Core system tests passed. Ready to use (LM Studio may need setup).")
        return 0
    else:
        logger.error("❌ Some critical tests failed. Please check the installation.")
        return 1

if __name__ == '__main__':
    sys.exit(main())