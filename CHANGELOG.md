# Changelog

All notable changes to HybridRAG will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- SystemD service configuration for production deployment
- Service management script with comprehensive controls
- Enhanced medical entity extraction using spaCy SciBERT
- Semantic chunking with medical content optimization
- S-PubMedBert embeddings for medical literature
- Directory batch processing via CLI
- Real-time knowledge graph metrics
- Production-ready logging and monitoring

### Changed
- Upgraded to FastAPI from Flask for better performance
- Enhanced document processor with multi-format support
- Improved medical entity recognition patterns
- Updated dependency management with separate environments
- Optimized knowledge graph construction for medical relationships

### Fixed
- CLI directory processing functionality
- Virtual environment activation scripts
- Memory management for large document processing
- Service configuration warnings and deprecations

## [1.0.0] - 2025-10-04

### Added
- Initial release of HybridRAG Emergency Medicine Knowledge System
- Core hybrid RAG functionality combining vector search and knowledge graphs
- Docling integration for high-quality PDF processing
- Medical entity extraction and relationship mapping
- ChromaDB vector database with local storage
- NetworkX knowledge graph with medical ontology support
- Cross-encoder reranking for relevance optimization
- LM Studio integration for local LLM inference
- FastAPI web interface with interactive chat
- Command-line interface for all operations
- Comprehensive configuration management
- Medical terminology and clinical decision support
- Emergency medicine protocol optimization
- Real-time system status monitoring
- Document ingestion with OCR and table extraction
- Semantic search with medical context awareness

### Features
- **Document Processing**: Multi-format support (PDF, DOCX, PPTX, XLSX, CSV, MD, HTML)
- **Medical AI**: Specialized entity extraction for symptoms, conditions, medications, procedures, anatomy
- **Hybrid Search**: Vector similarity + knowledge graph traversal + reranking
- **Local Deployment**: No cloud dependencies, complete privacy
- **Production Ready**: SystemD service, logging, monitoring, resource limits
- **Web Interface**: Interactive chat with source attribution
- **CLI Tools**: Batch processing, status monitoring, direct queries
- **Extensible**: Plugin architecture for additional medical domains

### Technical Stack
- **Backend**: Python 3.8+, FastAPI, ChromaDB, NetworkX
- **AI/ML**: spaCy SciBERT, S-PubMedBert, cross-encoders, Docling
- **Frontend**: HTML5, CSS3, JavaScript, real-time WebSockets
- **Storage**: Local files, SQLite, pickle serialization
- **Deployment**: SystemD, virtual environments, service management
- **Documentation**: Comprehensive guides for setup, usage, and deployment

### Medical Focus
- **Emergency Medicine**: Optimized for ED workflows and protocols
- **Clinical Decision Support**: Evidence-based information retrieval
- **Medical Literature**: Specialized processing for research papers and guidelines
- **Safety First**: Appropriate disclaimers and accuracy validation
- **Healthcare Privacy**: Local processing with no external data transmission

---

### 📋 Version Notes

This project follows semantic versioning:
- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions  
- **PATCH** version for backwards-compatible bug fixes

### 🏥 Medical Disclaimer

All versions include appropriate medical disclaimers. This system is for educational and research purposes only and should not be used for direct patient care without proper validation by qualified healthcare professionals.