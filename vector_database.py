import logging
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path

from config import config

logger = logging.getLogger(__name__)

class VectorDatabase:
    """Vector database for storing and querying document embeddings"""
    
    def __init__(self):
        """Initialize ChromaDB and embedding model"""
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(config.embedding_model)
        logger.info(f"Loaded embedding model: {config.embedding_model}")
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=config.vector_db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(config.collection_name)
            logger.info(f"Loaded existing collection: {config.collection_name}")
        except Exception:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=config.collection_name,
                metadata={"description": "Emergency medicine documents"}
            )
            logger.info(f"Created new collection: {config.collection_name}")
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Add document chunks to the vector database
        
        Args:
            chunks: List of document chunks with text and metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not chunks:
                logger.warning("No chunks provided to add")
                return False
            
            # Extract texts and prepare data
            texts = [chunk['text'] for chunk in chunks]
            chunk_ids = [chunk['chunk_id'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]
            
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(
                texts, 
                convert_to_numpy=True,
                show_progress_bar=True
            ).tolist()
            
            # Add to ChromaDB collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=chunk_ids
            )
            
            logger.info(f"Successfully added {len(chunks)} chunks to vector database")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector database: {str(e)}")
            return False
    
    def query_similar(
        self, 
        query_text: str, 
        n_results: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query for similar documents
        
        Args:
            query_text: Text to search for
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of similar documents with scores and metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query_text]).tolist()
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                where=filter_metadata,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and len(results['documents']) > 0:
                for i in range(len(results['documents'][0])):
                    result = {
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'similarity_score': 1 - results['distances'][0][i]  # Convert distance to similarity
                    }
                    formatted_results.append(result)
            
            logger.info(f"Retrieved {len(formatted_results)} similar documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying vector database: {str(e)}")
            return []
    
    def query_by_metadata(
        self, 
        metadata_filter: Dict[str, Any], 
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Query documents by metadata only
        
        Args:
            metadata_filter: Metadata filter criteria
            n_results: Maximum number of results
            
        Returns:
            List of matching documents
        """
        try:
            results = self.collection.get(
                where=metadata_filter,
                limit=n_results,
                include=['documents', 'metadatas']
            )
            
            formatted_results = []
            if results['documents']:
                for i in range(len(results['documents'])):
                    result = {
                        'text': results['documents'][i],
                        'metadata': results['metadatas'][i],
                        'similarity_score': 1.0  # No similarity calculation for metadata query
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying by metadata: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'embedding_dimension': len(self.embedding_model.encode(['test'])[0]),
                'model_name': config.embedding_model
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}
    
    def delete_collection(self):
        """Delete the entire collection (use with caution)"""
        try:
            self.client.delete_collection(config.collection_name)
            logger.info(f"Deleted collection: {config.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
    
    def reset_collection(self):
        """Reset the collection (delete all documents)"""
        try:
            self.delete_collection()
            self.collection = self.client.create_collection(
                name=config.collection_name,
                metadata={"description": "Emergency medicine documents"}
            )
            logger.info(f"Reset collection: {config.collection_name}")
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
    
    def batch_add_documents(
        self, 
        chunks: List[Dict[str, Any]], 
        batch_size: int = 100
    ) -> bool:
        """
        Add documents in batches for large datasets
        
        Args:
            chunks: List of document chunks
            batch_size: Size of each batch
            
        Returns:
            True if all batches successful, False otherwise
        """
        try:
            total_chunks = len(chunks)
            logger.info(f"Adding {total_chunks} chunks in batches of {batch_size}")
            
            for i in range(0, total_chunks, batch_size):
                batch = chunks[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(total_chunks-1)//batch_size + 1}")
                
                if not self.add_documents(batch):
                    logger.error(f"Failed to add batch starting at index {i}")
                    return False
            
            logger.info(f"Successfully added all {total_chunks} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error in batch add: {str(e)}")
            return False