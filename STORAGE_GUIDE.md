# 🏥 Emergency Medicine Hybrid RAG System - Storage & Environment Guide

## 📊 Current System Status

### ✅ **Persistent Storage - YES!**
Your system DOES have persistent storage:

1. **Vector Database**: ChromaDB with SQLite backend
   - Location: `./vector_db/chroma.sqlite3` (160KB)
   - Stores: Document embeddings, metadata, search indices
   - Persistent: Survives restarts

2. **Knowledge Graph**: NetworkX with pickle serialization
   - Location: `./vector_db/knowledge_graph.pkl` (created on first use)
   - Stores: Medical entities and relationships
   - Persistent: Saved after each update

3. **Uploaded Files**: Temporary processing storage
   - Location: `./uploads/` (cleaned after processing)
   - Purpose: Secure file handling during upload

## 🔧 **Virtual Environment Status**
Currently running in: **miniconda base environment** (not isolated)

### Recommendation: Use Virtual Environment
```bash
# Create and activate virtual environment
source activate_system.sh

# Or manually:
python -m venv venv_docling
source venv_docling/bin/activate
pip install -r requirements.txt
```

## 📁 **Multiple File Upload - SUPPORTED!**
Your drag & drop interface already supports multiple files:

### ✅ Current Capabilities:
- **Drag multiple PDFs** into the drop zone
- **Select multiple files** with file browser
- **Batch processing** of all uploaded documents
- **Progress tracking** for each file
- **Error handling** for individual files
- **Automatic cleanup** after processing

### Usage:
1. **Drag & Drop**: Select multiple PDFs and drag to the drop zone
2. **File Browser**: Click "Choose Files" and select multiple PDFs
3. **Processing**: All files are processed in sequence
4. **Storage**: Each document is embedded and stored in ChromaDB
5. **Knowledge Graph**: Medical entities extracted and relationships mapped

## 🗄️ **Database Storage Details**

### Vector Database (ChromaDB)
```
Location: ./vector_db/chroma.sqlite3
Size: ~160KB (with test documents)
Contains:
├── Document embeddings (384-dimensional vectors)
├── Metadata (filename, chunk info, medical tags)
├── Search indices
└── Collection configuration
```

### Knowledge Graph (NetworkX + Pickle)
```
Location: ./vector_db/knowledge_graph.pkl
Contains:
├── Medical entities (symptoms, conditions, medications, procedures)
├── Relationships between entities
├── Entity confidence scores
└── Graph structure for traversal
```

### Configuration
```
Location: config.py
Settings:
├── Vector DB path: ./vector_db
├── Collection name: emergency_medicine
├── Embedding model: all-MiniLM-L6-v2
├── LM Studio: http://192.168.2.180:1234
└── Web interface: 0.0.0.0:5000
```

## 🚀 **Enhanced Usage Examples**

### Multiple Document Upload
1. **Web Interface**: Visit http://localhost:5000
2. **Drag Multiple Files**: Drop 5-10 emergency medicine PDFs at once
3. **Monitor Progress**: Watch each file get processed individually
4. **Chat with System**: Ask questions about all uploaded documents

### Persistent Storage Verification
```bash
# Check vector database
ls -lah vector_db/chroma.sqlite3

# Check knowledge graph
ls -lah vector_db/knowledge_graph.pkl

# View document count via API
curl http://localhost:5000/api/status
```

## 🔍 **Storage Benefits**
- **Incremental Updates**: Add new documents without losing existing ones
- **Fast Retrieval**: Indexed search across all documents
- **Relationship Mapping**: Medical entity connections preserved
- **Session Persistence**: Knowledge survives system restarts
- **Scalable**: Handles hundreds of documents efficiently

Your system is well-architected for persistent, scalable document storage with both vector embeddings and knowledge graph relationships!