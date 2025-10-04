# HybridRAG - Emergency Medicine Knowledge System

[![Python 3.10](https://img.shields.io/badge/python-3.10.x-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Latest-purple.svg)](https://www.trychroma.com/)

A sophisticated **Hybrid Retrieval-Augmented Generation (RAG)** system specifically designed for emergency medicine, combining vector search, knowledge graphs, and reranking for enhanced information retrieval from medical literature and documents.

> **🏥 Built for Emergency Medicine**: Optimized for medical terminology, clinical decision-making, and emergency care protocols with specialized entity extraction and relationship mapping.

## Features

### 🔍 **Hybrid Retrieval**
- **Vector Search**: Semantic similarity using sentence transformers
- **Knowledge Graph**: Entity relationships and medical concept connections
- **Reranking**: Cross-encoder models for improved relevance

### 📄 **Advanced Document Processing**
- **Docling Integration**: High-quality PDF processing with OCR and table extraction
- **Medical Entity Extraction**: Automatic identification of symptoms, conditions, medications, procedures, and anatomy
- **Structured Knowledge**: Relationship extraction between medical concepts

### 🧠 **Medical AI & NLP**
- **spaCy SciBERT**: Advanced medical entity extraction (requires Python 3.10.x)
- **S-PubMedBert Embeddings**: Medical literature optimized embeddings
- **LM Studio Support**: Connect to your local LM Studio instance
- **Medical-Focused Prompts**: Specialized system prompts for emergency medicine
- **Context-Aware Responses**: Incorporates retrieved documents for accurate answers

### 🌐 **Web Interface**
- **Interactive Chat**: Beautiful web-based chat interface
- **Real-time Search**: Configurable search parameters
- **Source Transparency**: See which documents contributed to each answer
- **System Monitoring**: Live status of all components

### 🗃️ **Local Data Storage**
- **ChromaDB**: Local vector database for embeddings
- **NetworkX/Neo4j**: Knowledge graph storage options
- **No Cloud Dependencies**: Everything runs locally for privacy

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/tourniquetrules/HybridRAG.git
cd HybridRAG
chmod +x setup.sh
./setup.sh
```

> **⚠️ Important**: **Python 3.10.x is required** for spaCy SciBERT medical models to install correctly. The setup script will detect and use Python 3.10 automatically.

### 2. Activate Environment
```bash
# Using Python 3.10 activation script (recommended)
source activate_py310.sh

# Or activate directly
source venv_py310/bin/activate
```

### 3. Prepare Documents
```bash
# Place your emergency medicine PDFs in the documents folder
mkdir -p documents
# Copy your PDF files to documents/
```

### 4. Ingest Documents
```bash
# Ingest all PDFs in the documents folder
python cli.py ingest documents/

# Or with database reset
python cli.py ingest documents/ --reset
```

### 5. Install Medical Models (Automatic)
The setup script automatically installs spaCy SciBERT for medical entity extraction:
```bash
# This is done automatically by setup.sh, but can be run manually:
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz
```

> **🚀 GPU Acceleration**: For optimal performance with spaCy SciBERT, configure GPU support following the [spaCy GPU setup guide](https://spacy.io/usage#gpu).

### 6. Start the System

#### Development Mode
```bash
# Start the web server
python fastapi_app.py

# Access at http://localhost:5000
```

#### Production Mode (SystemD Service)
```bash
# Install as system service
sudo ./service-manager.sh install
sudo ./service-manager.sh start

# Check status
./service-manager.sh status

# Access at http://localhost:5000
```

## Usage Options

### Web Interface
The web interface provides a complete chat experience with:
- Real-time search configuration
- Source document display
- System status monitoring
- Document ingestion controls

Access at: `http://localhost:5000`

### Command Line Interface

#### Interactive Chat
```bash
python cli.py chat
```

#### Direct Query
```bash
python cli.py query "What is the treatment for septic shock?"
```

#### System Status
```bash
python cli.py status
```

#### Document Ingestion
```bash
# Basic ingestion
python cli.py ingest /path/to/pdfs/

# With options
python cli.py ingest /path/to/pdfs/ --reset --neo4j
```

## Configuration

### Environment Variables (.env)
```bash
# LM Studio Configuration
LM_STUDIO_HOST=http://192.168.2.64:1234
LM_STUDIO_MODEL=local-model

# Vector Database
VECTOR_DB_PATH=./vector_db
COLLECTION_NAME=emergency_medicine

# Knowledge Graph (if using Neo4j)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Models
EMBEDDING_MODEL=all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Web Server
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=True

# Processing
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MAX_DOCUMENTS=1000
```

## API Endpoints

### POST /api/chat
Chat with the RAG system
```json
{
  "query": "What are the signs of cardiac arrest?",
  "use_vector": true,
  "use_knowledge_graph": true,
  "use_reranking": true,
  "max_results": 10
}
```

### POST /api/search
Direct search without LLM response
```json
{
  "query": "epinephrine dosage",
  "use_vector": true,
  "use_knowledge_graph": true,
  "use_reranking": true,
  "max_results": 5
}
```

### POST /api/ingest
Ingest new documents
```json
{
  "documents_path": "/path/to/documents",
  "reset_db": false
}
```

### GET /api/status
Get system status
```json
{
  "vector_database": {
    "status": "online",
    "total_documents": 1500,
    "embedding_dimension": 384
  },
  "knowledge_graph": {
    "status": "online",
    "type": "NetworkX",
    "nodes": 2500,
    "relationships": 5000
  },
  "reranker": {
    "status": "online",
    "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
  },
  "llm": {
    "status": "online",
    "endpoint": "http://192.168.2.64:1234"
  }
}
```

## Architecture

### Components

1. **Document Processor** (`document_processor.py`)
   - Docling integration for PDF processing
   - Text chunking with overlap
   - Table extraction and formatting

2. **Vector Database** (`vector_database.py`)
   - ChromaDB for local storage
   - Sentence transformer embeddings
   - Similarity search and filtering

3. **Knowledge Graph** (`knowledge_graph.py`)
   - Medical entity extraction
   - Relationship identification
   - NetworkX or Neo4j storage

4. **Reranker** (`reranker.py`)
   - Cross-encoder relevance scoring
   - Result filtering and ranking

5. **LLM Client** (`llm_client.py`)
   - LM Studio API integration
   - Medical-focused prompting
   - Context-aware generation

6. **Hybrid RAG** (`hybrid_rag.py`)
   - Orchestrates all components
   - Implements hybrid search strategy
   - Handles result fusion

### Search Strategy

1. **Vector Search**: Find semantically similar document chunks
2. **Knowledge Graph Search**: Identify related medical entities and their connections
3. **Result Fusion**: Combine and deduplicate results from both methods
4. **Reranking**: Apply cross-encoder scoring for final relevance ranking
5. **LLM Generation**: Generate contextual response using top-ranked results

## Requirements

### System Requirements
- **Python 3.10.x** (Required for spaCy SciBERT medical models)
- **GPU**: CUDA-compatible GPU recommended for spaCy SciBERT (see [spaCy GPU setup](https://spacy.io/usage#gpu))
- **Memory**: 4GB+ RAM (8GB+ recommended for large document collections)
- **Storage**: 10GB+ free disk space
- **OS**: Linux, macOS, or Windows
- **Additional**: tesseract-ocr, poppler-utils (for PDF processing)

### LM Studio Setup
1. Install LM Studio
2. Load your preferred medical/general LLM
3. Start the server on `http://192.168.2.64:1234`
4. Ensure the model is loaded and ready

### Optional: Neo4j Setup
For enhanced knowledge graph capabilities:
```bash
# Install Neo4j
# Ubuntu/Debian
sudo apt-get install neo4j

# macOS
brew install neo4j

# Start Neo4j
sudo systemctl start neo4j
# or
neo4j start

# Access browser interface at http://localhost:7474
# Default credentials: neo4j/neo4j (change on first login)
```

## Medical Entity Types

The knowledge graph automatically extracts and relates:

- **Symptoms**: chest pain, dyspnea, fever, syncope, etc.
- **Conditions**: MI, stroke, sepsis, anaphylaxis, etc.
- **Medications**: epinephrine, atropine, morphine, etc.
- **Procedures**: intubation, defibrillation, central line, etc.
- **Anatomy**: heart, lung, aorta, ventricle, etc.

## Customization

### Adding Medical Patterns
Edit `knowledge_graph.py` to add domain-specific entity patterns:

```python
self.medical_patterns = {
    'symptoms': ['your_custom_symptoms'],
    'conditions': ['your_custom_conditions'],
    # ...
}
```

### Custom Models
Update `.env` file to use different models:
```bash
EMBEDDING_MODEL=your-embedding-model
RERANKER_MODEL=your-reranker-model
```

### Search Tuning
Adjust search parameters in the web interface or via API:
- `chunk_size`: Document chunk size (default: 512)
- `chunk_overlap`: Overlap between chunks (default: 50)
- `max_results`: Maximum search results (default: 10)

## Troubleshooting

### Common Issues

1. **LM Studio Connection Failed**
   - Ensure LM Studio is running at the correct address
   - Check firewall settings
   - Verify model is loaded

2. **spaCy Model Missing**
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **spaCy SciBERT Installation Issues**
   ```bash
   # Manual installation if setup.sh fails
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz
   
   # Verify installation
   python -c "import spacy; nlp = spacy.load('en_core_sci_scibert'); print('SciBERT loaded successfully')"
   ```
   - Requires Python 3.10.x
   - GPU recommended for optimal performance ([spaCy GPU guide](https://spacy.io/usage#gpu))
   - Large model (~1.2GB download)

4. **Memory Issues**
   - Reduce `chunk_size` and `max_results`
   - Process documents in smaller batches
   - Use lighter embedding models

4. **Neo4j Connection Issues**
   - Check if Neo4j is running: `sudo systemctl status neo4j`
   - Verify credentials in `.env`
   - Fall back to NetworkX: remove `--neo4j` flag

### Logging
Logs are written to console. For file logging, modify the logging configuration in each module.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Your License Here]

## Support

For issues and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the logs for error details

---

**Note**: This system is designed for educational and research purposes. Always consult current medical guidelines and qualified healthcare professionals for patient care decisions.