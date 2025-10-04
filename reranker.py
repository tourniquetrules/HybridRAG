import logging
from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder
import numpy as np

from config import config

logger = logging.getLogger(__name__)

class Reranker:
    """Reranks retrieved documents for improved relevance"""
    
    def __init__(self):
        """Initialize the cross-encoder model for reranking"""
        try:
            self.model = CrossEncoder(config.reranker_model)
            logger.info(f"Loaded reranker model: {config.reranker_model}")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {str(e)}")
            self.model = None
    
    def rerank_documents(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on query relevance
        
        Args:
            query: Search query
            documents: List of documents with text and metadata
            top_k: Number of top documents to return (if None, returns all)
            
        Returns:
            Reranked list of documents with rerank scores
        """
        if not self.model or not documents:
            logger.warning("Reranker model not available or no documents provided")
            return documents[:top_k] if top_k else documents
        
        try:
            # Prepare query-document pairs
            pairs = []
            for doc in documents:
                pairs.append([query, doc['text']])
            
            # Get rerank scores
            logger.info(f"Reranking {len(documents)} documents...")
            scores = self.model.predict(pairs)
            
            # Add rerank scores to documents
            for i, doc in enumerate(documents):
                doc['rerank_score'] = float(scores[i])
            
            # Sort by rerank score (highest first)
            reranked_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
            
            # Return top_k if specified
            if top_k:
                reranked_docs = reranked_docs[:top_k]
            
            logger.info(f"Reranking complete, returning {len(reranked_docs)} documents")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            # Fall back to original order
            return documents[:top_k] if top_k else documents
    
    def get_relevance_scores(self, query: str, texts: List[str]) -> List[float]:
        """
        Get relevance scores for a list of texts
        
        Args:
            query: Search query
            texts: List of text strings
            
        Returns:
            List of relevance scores
        """
        if not self.model:
            return [0.0] * len(texts)
        
        try:
            pairs = [[query, text] for text in texts]
            scores = self.model.predict(pairs)
            return scores.tolist()
        except Exception as e:
            logger.error(f"Error getting relevance scores: {str(e)}")
            return [0.0] * len(texts)
    
    def batch_rerank(
        self, 
        queries: List[str], 
        document_sets: List[List[Dict[str, Any]]], 
        top_k: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """
        Rerank multiple sets of documents for multiple queries
        
        Args:
            queries: List of search queries
            document_sets: List of document lists (one per query)
            top_k: Number of top documents per query
            
        Returns:
            List of reranked document lists
        """
        if len(queries) != len(document_sets):
            raise ValueError("Number of queries must match number of document sets")
        
        results = []
        for query, docs in zip(queries, document_sets):
            reranked = self.rerank_documents(query, docs, top_k)
            results.append(reranked)
        
        return results
    
    def compare_documents(self, query: str, doc1: str, doc2: str) -> Dict[str, Any]:
        """
        Compare two documents for relevance to a query
        
        Args:
            query: Search query
            doc1: First document text
            doc2: Second document text
            
        Returns:
            Comparison result with scores and winner
        """
        if not self.model:
            return {
                'doc1_score': 0.0,
                'doc2_score': 0.0,
                'winner': 'tie',
                'confidence': 0.0
            }
        
        try:
            scores = self.get_relevance_scores(query, [doc1, doc2])
            doc1_score = scores[0]
            doc2_score = scores[1]
            
            if doc1_score > doc2_score:
                winner = 'doc1'
                confidence = doc1_score - doc2_score
            elif doc2_score > doc1_score:
                winner = 'doc2'
                confidence = doc2_score - doc1_score
            else:
                winner = 'tie'
                confidence = 0.0
            
            return {
                'doc1_score': float(doc1_score),
                'doc2_score': float(doc2_score),
                'winner': winner,
                'confidence': float(confidence)
            }
            
        except Exception as e:
            logger.error(f"Error comparing documents: {str(e)}")
            return {
                'doc1_score': 0.0,
                'doc2_score': 0.0,
                'winner': 'error',
                'confidence': 0.0
            }
    
    def filter_by_threshold(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Filter documents by relevance threshold
        
        Args:
            query: Search query
            documents: List of documents
            threshold: Minimum relevance score
            
        Returns:
            Filtered list of documents above threshold
        """
        if not self.model:
            return documents
        
        reranked = self.rerank_documents(query, documents)
        filtered = [doc for doc in reranked if doc.get('rerank_score', 0) >= threshold]
        
        logger.info(f"Filtered {len(documents)} documents to {len(filtered)} above threshold {threshold}")
        return filtered