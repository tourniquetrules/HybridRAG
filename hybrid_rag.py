import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import asyncio

from enhanced_document_processor import EnhancedDocumentProcessor as DocumentProcessor
from vector_database import VectorDatabase
from knowledge_graph import MedicalKnowledgeGraph
from reranker import Reranker
from llm_client import LMStudioClient
from config import config

logger = logging.getLogger(__name__)

class HybridRAGSystem:
    """Hybrid RAG system combining vector search, knowledge graphs, and reranking"""
    
    def __init__(self, use_neo4j: bool = False):
        """
        Initialize the Hybrid RAG system
        
        Args:
            use_neo4j: Whether to use Neo4j for knowledge graph (requires Neo4j installation)
        """
        logger.info("Initializing Hybrid RAG System...")
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.vector_db = VectorDatabase()
        self.knowledge_graph = MedicalKnowledgeGraph(use_neo4j=use_neo4j)
        
        # Setup dedicated document processing log
        self.setup_document_log()
        
        # Load existing knowledge graph if it exists
        if not self.knowledge_graph.use_neo4j:
            graph_path = Path(config.vector_db_path) / "knowledge_graph.pkl"
            if graph_path.exists():
                self.knowledge_graph.load_graph(str(graph_path))
                logger.info(f"Loaded existing knowledge graph from {graph_path}")
        
        self.reranker = Reranker()
        self.llm_client = LMStudioClient()
        
        logger.info("Hybrid RAG System initialized successfully")
    
    def setup_document_log(self):
        """Setup dedicated document processing log file"""
        from datetime import datetime
        
        self.doc_log_file = Path("logs") / "document_processing.log"
        self.doc_log_file.parent.mkdir(exist_ok=True)
        
        # Setup dedicated logger for document processing
        self.doc_logger = logging.getLogger("document_processing")
        self.doc_logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.doc_logger.handlers[:]:
            self.doc_logger.removeHandler(handler)
        
        # Create file handler
        handler = logging.FileHandler(self.doc_log_file)
        handler.setLevel(logging.INFO)
        
        # Simple format for document log
        formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        
        self.doc_logger.addHandler(handler)
        self.doc_logger.propagate = False  # Don't propagate to root logger
        
        # Write header if new file
        if not self.doc_log_file.exists() or self.doc_log_file.stat().st_size == 0:
            self.doc_logger.info("=" * 80)
            self.doc_logger.info("DOCUMENT PROCESSING LOG")
            self.doc_logger.info("=" * 80)
            self.doc_logger.info("Format: TIMESTAMP | DOCUMENT_NAME | CHUNKED: YES/NO (count) | KNOWLEDGE_GRAPH: YES/NO (nodes+relationships)")
    
    def log_document_processing(self, doc_name: str, chunks_created: int, 
                              kg_nodes_added: int, kg_relationships_added: int, 
                              success: bool = True):
        """Log document processing result to dedicated log file"""
        try:
            # Determine chunking status
            chunked_status = f"YES ({chunks_created})" if chunks_created > 0 else "NO (0)"
            
            # Determine knowledge graph status
            kg_status = f"YES (+{kg_nodes_added} nodes, +{kg_relationships_added} rels)" if (kg_nodes_added > 0 or kg_relationships_added > 0) else "NO (no entities)"
            
            # Overall status
            status_emoji = "✅" if success else "❌"
            
            # Log entry
            log_entry = f"{status_emoji} {doc_name} | CHUNKED: {chunked_status} | KNOWLEDGE_GRAPH: {kg_status}"
            self.doc_logger.info(log_entry)
            
        except Exception as e:
            logger.error(f"Error logging document processing: {str(e)}")
    
    def ingest_documents(self, documents_path: str, reset_db: bool = False) -> bool:
        """
        Ingest documents from a directory
        
        Args:
            documents_path: Path to directory containing PDF documents
            reset_db: Whether to reset existing databases
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if reset_db:
                logger.info("Resetting databases...")
                self.vector_db.reset_collection()
                # Note: Knowledge graph reset would need to be implemented based on storage type
            
            logger.info(f"Starting document ingestion from: {documents_path}")
            
            # Process documents with Docling
            all_chunks = asyncio.run(self.document_processor.process_directory(documents_path))
            
            if not all_chunks:
                logger.warning("No chunks extracted from documents")
                return False
            
            logger.info(f"Extracted {len(all_chunks)} chunks from documents")
            
            # Add to vector database
            logger.info("Adding chunks to vector database...")
            if not self.vector_db.batch_add_documents(all_chunks, batch_size=50):
                logger.error("Failed to add documents to vector database")
                return False
            
            # Extract entities and build knowledge graph
            logger.info("Building knowledge graph...")
            self._build_knowledge_graph(all_chunks)
            
            # Save knowledge graph if using NetworkX
            if not self.knowledge_graph.use_neo4j:
                graph_path = Path(config.vector_db_path) / "knowledge_graph.pkl"
                self.knowledge_graph.save_graph(str(graph_path))
            
            logger.info("Document ingestion completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during document ingestion: {str(e)}")
            return False
    
    def ingest_single_document(self, document_path: str) -> bool:
        """
        Ingest a single document file
        
        Args:
            document_path: Path to a single PDF document
            
        Returns:
            True if successful, False otherwise
        """
        # Extract clean document name from path
        doc_name = Path(document_path).name
        
        # Initialize tracking variables
        chunks_created = 0
        vector_db_success = False
        kg_nodes_before = 0
        kg_edges_before = 0
        kg_nodes_after = 0
        kg_edges_after = 0
        
        try:
            logger.info(f"📄 DOCUMENT PROCESSING START: {doc_name}")
            logger.info(f"Processing single document: {document_path}")
            
            # Get initial knowledge graph state
            if self.knowledge_graph and hasattr(self.knowledge_graph, 'graph'):
                kg_nodes_before = len(self.knowledge_graph.graph.nodes())
                kg_edges_before = len(self.knowledge_graph.graph.edges())
            
            # Process single document with Docling
            result = self.document_processor.process_document(document_path)
            
            if not result:
                logger.error(f"❌ DOCUMENT FAILED: {doc_name} - Docling conversion failed")
                return False
            
            logger.info(f"✅ Docling conversion: SUCCESS")
            
            # Extract chunks from the conversion result
            chunks = self.document_processor.extract_chunks(result)
            
            if not chunks:
                logger.error(f"❌ DOCUMENT FAILED: {doc_name} - No chunks extracted")
                return False
            
            chunks_created = len(chunks)
            logger.info(f"✅ Chunking: SUCCESS - Created {chunks_created} chunks")
            
            # Add to vector database
            logger.info("Adding chunks to vector database...")
            if not self.vector_db.batch_add_documents(chunks, batch_size=50):
                logger.error(f"❌ DOCUMENT FAILED: {doc_name} - Vector database ingestion failed")
                return False
            
            vector_db_success = True
            logger.info(f"✅ Vector Database: SUCCESS - {chunks_created} chunks stored")
            
            # Extract entities and build knowledge graph
            logger.info("Updating knowledge graph...")
            self._build_knowledge_graph(chunks)
            
            # Get final knowledge graph state
            if self.knowledge_graph and hasattr(self.knowledge_graph, 'graph'):
                kg_nodes_after = len(self.knowledge_graph.graph.nodes())
                kg_edges_after = len(self.knowledge_graph.graph.edges())
            
            nodes_added = kg_nodes_after - kg_nodes_before
            relationships_added = kg_edges_after - kg_edges_before
            
            logger.info(f"✅ Knowledge Graph: SUCCESS - Added {nodes_added} nodes, {relationships_added} relationships")
            
            # Save knowledge graph if using NetworkX
            if not self.knowledge_graph.use_neo4j:
                graph_path = Path(config.vector_db_path) / "knowledge_graph.pkl"
                self.knowledge_graph.save_graph(str(graph_path))
            
            # Final success summary
            logger.info(f"📊 DOCUMENT SUCCESS SUMMARY: {doc_name}")
            logger.info(f"   📄 Document: {doc_name}")
            logger.info(f"   📚 Chunks Created: {chunks_created}")
            logger.info(f"   💾 Vector Database: SUCCESS")
            logger.info(f"   🧠 Knowledge Graph Nodes: {kg_nodes_before} → {kg_nodes_after} (+{nodes_added})")
            logger.info(f"   🔗 Knowledge Graph Relationships: {kg_edges_before} → {kg_edges_after} (+{relationships_added})")
            logger.info(f"   ✅ Overall Status: COMPLETE")
            
            # Log to dedicated document log
            self.log_document_processing(doc_name, chunks_created, nodes_added, relationships_added, success=True)
            
            logger.info(f"Single document processing completed: {document_path}")
            return True
            
        except Exception as e:
            nodes_added = kg_nodes_after - kg_nodes_before
            relationships_added = kg_edges_after - kg_edges_before
            
            logger.error(f"❌ DOCUMENT ERROR: {doc_name}")
            logger.error(f"   📄 Document: {doc_name}")
            logger.error(f"   📚 Chunks Created: {chunks_created}")
            logger.error(f"   💾 Vector Database: {'SUCCESS' if vector_db_success else 'FAILED'}")
            logger.error(f"   🧠 Knowledge Graph Nodes: {kg_nodes_before} → {kg_nodes_after}")
            logger.error(f"   🔗 Knowledge Graph Relationships: {kg_edges_before} → {kg_edges_after}")
            logger.error(f"   ❌ Error: {str(e)}")
            
            # Log to dedicated document log (failed)
            self.log_document_processing(doc_name, chunks_created, nodes_added, relationships_added, success=False)
            
            logger.error(f"Error processing single document {document_path}: {str(e)}")
            return False
    
    def _build_knowledge_graph(self, chunks: List[Dict[str, Any]]):
        """Build knowledge graph from document chunks"""
        logger.info("Extracting entities and relationships...")
        
        for i, chunk in enumerate(chunks):
            if i % 100 == 0:
                logger.info(f"Processing chunk {i+1}/{len(chunks)} for knowledge graph")
            
            try:
                # Debug: Log chunk text preview
                text_preview = chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
                logger.debug(f"Processing chunk text: {text_preview}")
                
                # Extract entities
                entities = self.knowledge_graph.extract_entities(chunk['text'])
                
                # Debug: Log what was extracted
                if entities:
                    logger.info(f"Chunk {i+1}: Found {len(entities)} entity types: {list(entities.keys())}")
                    for entity_type, entity_list in entities.items():
                        logger.info(f"  {entity_type}: {entity_list}")
                else:
                    logger.info(f"Chunk {i+1}: No entities found")
                
                # Extract relationships
                relationships = self.knowledge_graph.extract_relationships(chunk['text'], entities)
                
                # Debug: Log relationships
                if relationships:
                    logger.info(f"Chunk {i+1}: Found {len(relationships)} relationships")
                    for rel in relationships[:5]:  # Log first 5
                        logger.info(f"  {rel[0]} --{rel[1]}--> {rel[2]}")
                else:
                    logger.info(f"Chunk {i+1}: No relationships found")
                
                # Add to knowledge graph
                self.knowledge_graph.add_to_graph(entities, relationships, chunk['chunk_id'])
                
            except Exception as e:
                logger.warning(f"Error processing chunk {chunk['chunk_id']} for KG: {str(e)}")
                import traceback
                logger.debug(f"KG processing error traceback: {traceback.format_exc()}")
                continue
        
        # Log statistics
        stats = self.knowledge_graph.get_graph_stats()
        logger.info(f"Knowledge graph built: {stats['nodes']} nodes, {stats['relationships']} relationships")
    
    def search(
        self, 
        query: str, 
        use_vector: bool = True,
        use_knowledge_graph: bool = True,
        use_reranking: bool = True,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector similarity and knowledge graph
        
        Args:
            query: Search query
            use_vector: Whether to use vector search
            use_knowledge_graph: Whether to use knowledge graph
            use_reranking: Whether to apply reranking
            max_results: Maximum number of results to return
            
        Returns:
            List of ranked search results
        """
        try:
            results = []
            
            # Vector search
            if use_vector:
                logger.info("Performing vector search...")
                vector_results = self.vector_db.query_similar(
                    query, 
                    n_results=max_results * 2  # Get more for reranking
                )
                
                # Add search method to metadata
                for result in vector_results:
                    result['search_method'] = 'vector'
                    result['original_rank'] = len(results)
                
                results.extend(vector_results)
            
            # Knowledge graph search
            if use_knowledge_graph:
                logger.info("Performing knowledge graph search...")
                kg_results = self._knowledge_graph_search(query, max_results)
                
                # Add search method to metadata
                for result in kg_results:
                    result['search_method'] = 'knowledge_graph'
                    result['original_rank'] = len(results)
                
                results.extend(kg_results)
            
            # Remove duplicates based on chunk_id
            results = self._deduplicate_results(results)
            
            # Rerank results
            if use_reranking and len(results) > 1:
                logger.info("Reranking results...")
                results = self.reranker.rerank_documents(
                    query, 
                    results, 
                    top_k=max_results
                )
            else:
                results = results[:max_results]
            
            logger.info(f"Search completed, returning {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return []
    
    def _knowledge_graph_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using knowledge graph"""
        results = []
        
        try:
            # Extract entities from query
            query_entities = self.knowledge_graph.extract_entities(query)
            
            # Find related entities for each query entity
            related_entities = set()
            for entity_type, entities in query_entities.items():
                for entity in entities:
                    related = self.knowledge_graph.query_related_entities(entity, max_depth=2)
                    for rel in related[:5]:  # Limit related entities per query entity
                        related_entities.add(rel['entity'])
            
            # Query vector database for documents containing related entities
            if related_entities:
                # Create search terms from related entities
                search_terms = list(related_entities)[:10]  # Limit search terms
                
                for term in search_terms:
                    # Search for documents containing this entity
                    term_results = self.vector_db.query_similar(term, n_results=3)
                    
                    # Add knowledge graph metadata
                    for result in term_results:
                        result['kg_entity'] = term
                        result['search_method'] = 'knowledge_graph'
                        # Boost similarity score for KG matches
                        result['similarity_score'] = result.get('similarity_score', 0) * 1.2
                    
                    results.extend(term_results)
            
            # Remove duplicates and limit results
            results = self._deduplicate_results(results)[:max_results]
            
        except Exception as e:
            logger.warning(f"Error in knowledge graph search: {str(e)}")
        
        return results
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on chunk_id or text similarity"""
        seen_ids = set()
        unique_results = []
        
        for result in results:
            # Try to get chunk_id from metadata
            chunk_id = result.get('metadata', {}).get('chunk_id')
            if not chunk_id:
                # Generate ID from text hash if no chunk_id
                chunk_id = hash(result.get('text', ''))
            
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_results.append(result)
        
        return unique_results
    
    def chat(
        self, 
        query: str, 
        conversation_history: List[Dict[str, str]] = None,
        search_kwargs: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Chat interface with RAG capabilities
        
        Args:
            query: User query
            conversation_history: Previous conversation messages
            search_kwargs: Additional search parameters
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Default search parameters
            if search_kwargs is None:
                search_kwargs = {}
            
            # Perform hybrid search
            search_results = self.search(query, **search_kwargs)
            
            # Generate response using LLM with context
            response = self.llm_client.generate_rag_response(
                query=query,
                context_documents=search_results,
                conversation_history=conversation_history
            )
            
            # Prepare response metadata
            response_data = {
                'response': response,
                'query': query,
                'sources': [
                    {
                        'text': result.get('text', '')[:200] + '...',
                        'source': result.get('metadata', {}).get('source', 'Unknown'),
                        'similarity_score': result.get('similarity_score', 0),
                        'rerank_score': result.get('rerank_score'),
                        'search_method': result.get('search_method', 'unknown')
                    }
                    for result in search_results[:5]
                ],
                'num_sources': len(search_results),
                'search_methods_used': list(set(r.get('search_method', 'unknown') for r in search_results))
            }
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return {
                'response': f"I apologize, but I encountered an error while processing your question: {str(e)}",
                'query': query,
                'sources': [],
                'num_sources': 0,
                'search_methods_used': []
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all system components"""
        try:
            vector_stats = self.vector_db.get_collection_stats()
            kg_stats = self.knowledge_graph.get_graph_stats()
            
            return {
                'vector_database': {
                    'status': 'online',
                    **vector_stats
                },
                'knowledge_graph': {
                    'status': 'online',
                    'type': 'Neo4j' if self.knowledge_graph.use_neo4j else 'NetworkX',
                    **kg_stats
                },
                'reranker': {
                    'status': 'online' if self.reranker.model else 'offline',
                    'model': config.reranker_model
                },
                'llm': {
                    'status': 'online',  # Assume online if client was created
                    'endpoint': config.lm_studio_host
                }
            }
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {'error': str(e)}
    
    def clear_database(self):
        """Clear all databases (vector DB and knowledge graph)"""
        try:
            logger.info("Clearing vector database...")
            self.vector_db.reset_collection()
            
            logger.info("Clearing knowledge graph...")
            self.knowledge_graph.clear_graph()
            
            logger.info("Databases cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing databases: {str(e)}")
            return False
    
    def get_document_processing_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of document processing status"""
        try:
            vector_stats = self.vector_db.get_collection_stats() if self.vector_db else {}
            kg_stats = self.knowledge_graph.get_graph_stats() if self.knowledge_graph else {}
            
            # Get document metadata from vector database
            documents_info = []
            if self.vector_db and hasattr(self.vector_db, 'collection'):
                try:
                    # Query all documents to get metadata
                    all_results = self.vector_db.collection.get()
                    
                    # Group by document filename
                    doc_groups = {}
                    for i, metadata in enumerate(all_results.get('metadatas', [])):
                        if metadata and 'filename' in metadata:
                            filename = metadata['filename']
                            if filename not in doc_groups:
                                doc_groups[filename] = {
                                    'chunks': 0,
                                    'first_seen': metadata.get('timestamp', 'unknown'),
                                    'chunk_types': set()
                                }
                            doc_groups[filename]['chunks'] += 1
                            if 'chunk_type' in metadata:
                                doc_groups[filename]['chunk_types'].add(metadata['chunk_type'])
                    
                    # Convert to list format
                    for filename, info in doc_groups.items():
                        documents_info.append({
                            'document_name': filename,
                            'chunks_created': info['chunks'],
                            'processing_date': info['first_seen'],
                            'chunk_types': list(info['chunk_types']),
                            'status': 'SUCCESS'
                        })
                        
                except Exception as e:
                    logger.error(f"Error getting document metadata: {str(e)}")
            
            summary = {
                'total_documents_processed': len(documents_info),
                'total_chunks': vector_stats.get('total_documents', 0),
                'knowledge_graph_nodes': kg_stats.get('total_nodes', 0),
                'knowledge_graph_relationships': kg_stats.get('total_relationships', 0),
                'documents': documents_info,
                'system_status': {
                    'vector_database': 'READY' if self.vector_db else 'NOT_INITIALIZED',
                    'knowledge_graph': 'READY' if self.knowledge_graph else 'NOT_INITIALIZED',
                    'document_processor': 'READY' if self.document_processor else 'NOT_INITIALIZED'
                }
            }
            
            # Log the summary
            logger.info("📊 DOCUMENT PROCESSING SUMMARY")
            logger.info(f"   📄 Total Documents: {len(documents_info)}")
            logger.info(f"   📚 Total Chunks: {vector_stats.get('total_documents', 0)}")
            logger.info(f"   🧠 Knowledge Graph Nodes: {kg_stats.get('total_nodes', 0)}")
            logger.info(f"   🔗 Knowledge Graph Relationships: {kg_stats.get('total_relationships', 0)}")
            
            if documents_info:
                logger.info("   📋 Document Details:")
                for doc in documents_info:
                    logger.info(f"     • {doc['document_name']}: {doc['chunks_created']} chunks, {doc['status']}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting document processing summary: {str(e)}")
            return {'error': str(e)}
    
    def close(self):
        """Close all system components"""
        try:
            if hasattr(self.knowledge_graph, 'close'):
                self.knowledge_graph.close()
            logger.info("Hybrid RAG system closed")
        except Exception as e:
            logger.error(f"Error closing system: {str(e)}")