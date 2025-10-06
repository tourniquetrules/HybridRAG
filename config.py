import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # LM Studio Configuration
    lm_studio_host: str = os.getenv("LM_STUDIO_HOST", "http://192.168.2.180:1234")
    lm_studio_model: str = os.getenv("LM_STUDIO_MODEL", "local-model")
    
    # Vector Database Configuration
    vector_db_path: str = os.getenv("VECTOR_DB_PATH", "./vector_db")
    collection_name: str = os.getenv("COLLECTION_NAME", "emergency_medicine")
    
    # Knowledge Graph Configuration
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")
    
    # Embedding Models - Using scientific/medical models
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO")
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Web Interface
    flask_host: str = os.getenv("FLASK_HOST", "0.0.0.0")
    flask_port: int = int(os.getenv("FLASK_PORT", "5000"))
    flask_debug: bool = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    
    # Document Processing
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    max_documents: int = int(os.getenv("MAX_DOCUMENTS", "1000"))

# Global config instance
config = Config()