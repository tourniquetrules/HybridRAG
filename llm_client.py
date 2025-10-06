import logging
import requests
import json
from typing import List, Dict, Any, Optional, Generator
import time

from config import config

logger = logging.getLogger(__name__)

class LMStudioClient:
    """Client for interacting with LM Studio API"""
    
    def __init__(self):
        """Initialize LM Studio client"""
        self.base_url = config.lm_studio_host.rstrip('/')
        self.model = config.lm_studio_model
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Test connection and auto-detect model if needed
        self._test_connection()
        self._auto_detect_model()
    
    def _test_connection(self) -> bool:
        """Test connection to LM Studio"""
        try:
            response = self.session.get(f"{self.base_url}/v1/models", timeout=10)
            if response.status_code == 200:
                models = response.json()
                logger.info(f"Connected to LM Studio. Available models: {len(models.get('data', []))}")
                return True
            else:
                logger.warning(f"LM Studio connection test failed: {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"Cannot connect to LM Studio: {str(e)}")
            return False
    
    def _auto_detect_model(self) -> None:
        """Auto-detect available model if default doesn't work"""
        try:
            # First, try to get available models
            response = self.session.get(f"{self.base_url}/v1/models", timeout=10)
            if response.status_code == 200:
                models = response.json()
                available_models = [model['id'] for model in models.get('data', [])]
                
                if available_models:
                    # Use the first available model if current model is generic
                    if self.model in ["local-model", ""] or self.model not in available_models:
                        self.model = available_models[0]
                        logger.info(f"Auto-detected model: {self.model}")
                    else:
                        logger.info(f"Using configured model: {self.model}")
                else:
                    logger.warning("No models available in LM Studio")
            else:
                logger.warning("Could not retrieve model list from LM Studio")
        except Exception as e:
            logger.warning(f"Model auto-detection failed: {e}")
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from LM Studio"""
        try:
            response = self.session.get(f"{self.base_url}/v1/models", timeout=10)
            if response.status_code == 200:
                models = response.json()
                return models.get('data', [])
            else:
                logger.warning(f"Could not retrieve model list: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different model"""
        try:
            # Verify the model exists in available models
            available_models = self.get_available_models()
            available_model_ids = [model['id'] for model in available_models]
            
            if model_name in available_model_ids:
                self.model = model_name
                logger.info(f"Switched to model: {self.model}")
                return True
            else:
                logger.warning(f"Model {model_name} not available. Available models: {available_model_ids}")
                return False
        except Exception as e:
            logger.error(f"Error switching model: {e}")
            return False
    
    def get_current_model(self) -> str:
        """Get currently selected model"""
        return self.model
    
    def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> Optional[str]:
        """
        Generate response using LM Studio
        
        Args:
            messages: List of messages in OpenAI chat format
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Generated response text or None if failed
        """
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }
            
            if stream:
                return self._stream_response(payload)
            else:
                response = self.session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data['choices'][0]['message']['content']
                else:
                    logger.error(f"LM Studio API error: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return None
    
    def _stream_response(self, payload: Dict[str, Any]) -> Generator[str, None, None]:
        """Stream response from LM Studio"""
        try:
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=60
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            line = line[6:]
                            if line.strip() == '[DONE]':
                                break
                            try:
                                data = json.loads(line)
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        yield delta['content']
                            except json.JSONDecodeError:
                                continue
            else:
                logger.error(f"Streaming error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}")
    
    def generate_rag_response(
        self, 
        query: str, 
        context_documents: List[Dict[str, Any]], 
        conversation_history: List[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Generate RAG response with context
        
        Args:
            query: User query
            context_documents: Retrieved documents for context
            conversation_history: Previous conversation messages
            
        Returns:
            Generated response incorporating the context
        """
        # Prepare context from documents
        context_text = self._format_context(context_documents)
        
        # Create system prompt for emergency medicine
        system_prompt = self._get_system_prompt()
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history[-10:])  # Keep last 10 messages
        
        # Add current query with context
        user_message = self._format_user_message(query, context_text)
        messages.append({"role": "user", "content": user_message})
        
        return self.generate_response(messages, temperature=0.3, max_tokens=2000)
    
    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents as context"""
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(documents[:5]):  # Limit to top 5 documents
            source = doc.get('metadata', {}).get('source', 'Unknown')
            text = doc.get('text', '').strip()
            
            # Add rerank score if available
            score_info = ""
            if 'rerank_score' in doc:
                score_info = f" (Relevance: {doc['rerank_score']:.3f})"
            elif 'similarity_score' in doc:
                score_info = f" (Similarity: {doc['similarity_score']:.3f})"
            
            context_parts.append(f"Document {i+1} - {source}{score_info}:\n{text}")
        
        return "\n\n".join(context_parts)
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for emergency medicine assistant"""
        return """You are an expert emergency medicine assistant. You provide accurate, evidence-based medical information based on the provided context documents.

Guidelines:
- Always base your responses on the provided context documents
- If the context doesn't contain relevant information, clearly state this
- Provide specific medical details when available
- Include dosages, contraindications, and clinical pearls when mentioned in the context
- Always recommend consulting with medical professionals for actual patient care
- Be concise but thorough in your explanations
- If asked about treatments or procedures, mention both indications and potential complications

Important: This is for educational purposes only. Always defer to current clinical guidelines and consult with qualified medical professionals for patient care decisions."""
    
    def _format_user_message(self, query: str, context: str) -> str:
        """Format user message with context"""
        return f"""Context Documents:
{context}

User Question: {query}

Please provide a comprehensive answer based on the context documents above. If the context doesn't contain sufficient information to answer the question, please state this clearly."""
    
    def extract_entities_with_llm(self, text: str) -> List[str]:
        """Use LLM to extract medical entities from text"""
        messages = [{
            "role": "system",
            "content": "You are an expert at extracting medical entities from text. Extract all relevant medical terms including symptoms, conditions, medications, procedures, and anatomical terms."
        }, {
            "role": "user", 
            "content": f"Extract medical entities from this text. Return only a JSON list of entities:\n\n{text}"
        }]
        
        response = self.generate_response(messages, temperature=0.1, max_tokens=500)
        
        try:
            # Try to parse JSON response
            import json
            entities = json.loads(response)
            return entities if isinstance(entities, list) else []
        except:
            # Fallback: extract from text response
            if response:
                # Simple extraction of comma-separated entities
                entities = [entity.strip() for entity in response.split(',')]
                return [e for e in entities if e and not e.startswith('[') and not e.endswith(']')]
            return []
    
    def summarize_document(self, document_text: str, max_length: int = 500) -> Optional[str]:
        """Generate a summary of a document"""
        messages = [{
            "role": "system",
            "content": "You are an expert at summarizing medical documents. Create concise, accurate summaries that capture the key medical information."
        }, {
            "role": "user",
            "content": f"Summarize this medical document in {max_length} words or less:\n\n{document_text}"
        }]
        
        return self.generate_response(messages, temperature=0.3, max_tokens=max_length)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        try:
            response = self.session.get(f"{self.base_url}/v1/models", timeout=10)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {}