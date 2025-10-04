# Contributing to HybridRAG

Thank you for your interest in contributing to HybridRAG! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Issues

1. **Search existing issues** first to avoid duplicates
2. **Use the issue template** when creating new issues
3. **Provide detailed information**:
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version)
   - Relevant logs or error messages

### Feature Requests

1. **Check if the feature already exists** or is planned
2. **Describe the use case** clearly
3. **Explain the benefit** to emergency medicine workflows
4. **Consider implementation complexity** and maintainability

### Code Contributions

#### Prerequisites

- Python 3.8+
- Git knowledge
- Understanding of RAG systems and/or medical informatics
- Familiarity with FastAPI, ChromaDB, and spaCy

#### Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/[your-username]/HybridRAG.git
   cd HybridRAG
   ```

2. **Set up development environment**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   source activate_py310.sh
   ```

3. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

#### Code Standards

1. **Follow PEP 8** style guidelines
2. **Add docstrings** to all functions and classes
3. **Include type hints** where appropriate
4. **Write comprehensive tests** for new functionality
5. **Update documentation** for user-facing changes

#### Testing

1. **Run existing tests**
   ```bash
   python test_system.py
   ```

2. **Add tests for new features**
   - Unit tests for individual functions
   - Integration tests for workflows
   - Medical accuracy tests for domain-specific features

3. **Test medical terminology extraction**
   ```bash
   python test_fix.py
   ```

#### Medical Domain Considerations

When contributing medical-related features:

1. **Validate medical accuracy** with healthcare professionals
2. **Use established medical ontologies** (SNOMED CT, ICD-10, etc.)
3. **Follow emergency medicine guidelines** (AHA, ECC, etc.)
4. **Include appropriate disclaimers** for clinical features
5. **Test with real medical literature** when possible

#### Pull Request Process

1. **Ensure all tests pass**
2. **Update documentation** if needed
3. **Add changelog entry** for significant changes
4. **Fill out PR template** completely
5. **Request review** from maintainers

#### Code Review Criteria

- **Functionality**: Does it work as intended?
- **Medical accuracy**: Is medical information correct?
- **Performance**: Does it maintain system performance?
- **Security**: Are there any security implications?
- **Maintainability**: Is the code clean and well-documented?

## 📋 Development Guidelines

### File Organization

```
HybridRAG/
├── core/                   # Core RAG functionality
│   ├── hybrid_rag.py      # Main orchestrator
│   ├── vector_database.py # Vector storage
│   └── knowledge_graph.py # Medical knowledge graph
├── processors/             # Document processing
│   ├── document_processor.py
│   └── enhanced_document_processor.py
├── api/                   # Web API
│   ├── fastapi_app.py     # Main application
│   └── cli.py             # Command-line interface
├── config/                # Configuration
│   └── config.py          # Settings management
├── templates/             # Web templates
├── docs/                  # Documentation
└── tests/                 # Test suite
```

### Coding Conventions

#### Python Style
```python
# Good: Clear, descriptive names
def extract_medical_entities(text: str) -> Dict[str, List[str]]:
    """Extract medical entities from clinical text.
    
    Args:
        text: Input clinical text
        
    Returns:
        Dictionary mapping entity types to lists of entities
    """
    pass

# Good: Type hints and error handling
async def process_documents(
    documents_path: Path,
    reset_db: bool = False
) -> ProcessingResult:
    try:
        # Implementation
        pass
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise ProcessingError(f"Failed to process documents: {e}")
```

#### Medical Terminology
```python
# Good: Use standard medical abbreviations
MEDICAL_ENTITIES = {
    'symptoms': ['chest_pain', 'dyspnea', 'syncope'],
    'conditions': ['stemi', 'nstemi', 'pe', 'dvt'],
    'medications': ['epinephrine', 'atropine', 'amiodarone']
}

# Good: Include medical context
def calculate_risk_score(
    age: int,
    symptoms: List[str],
    vital_signs: VitalSigns
) -> RiskAssessment:
    """Calculate emergency department risk assessment."""
    pass
```

### Testing Guidelines

#### Unit Tests
```python
import pytest
from hybrid_rag import HybridRAGSystem

def test_medical_entity_extraction():
    """Test extraction of medical entities from clinical text."""
    text = "Patient presents with chest pain and dyspnea"
    entities = extract_medical_entities(text)
    
    assert 'symptoms' in entities
    assert 'chest_pain' in entities['symptoms']
    assert 'dyspnea' in entities['symptoms']

def test_emergency_protocol_retrieval():
    """Test retrieval of emergency protocols."""
    query = "cardiac arrest protocol"
    results = rag_system.search(query, max_results=5)
    
    assert len(results) > 0
    assert any('cpr' in r.content.lower() for r in results)
```

#### Integration Tests
```python
def test_end_to_end_medical_workflow():
    """Test complete medical document processing workflow."""
    # Setup test documents
    test_docs = create_test_medical_documents()
    
    # Process documents
    rag_system.ingest_documents(test_docs)
    
    # Test queries
    response = rag_system.query("What is the treatment for septic shock?")
    
    assert response.confidence > 0.7
    assert 'fluid resuscitation' in response.content.lower()
```

### Documentation

#### Code Documentation
- Use Google-style docstrings
- Include examples for complex functions
- Document medical assumptions and limitations
- Reference medical guidelines when applicable

#### User Documentation
- Update README.md for new features
- Add usage examples
- Include troubleshooting information
- Document medical disclaimers

## 🏥 Medical Content Guidelines

### Sources
- Use peer-reviewed medical literature
- Reference current clinical guidelines
- Include emergency medicine protocols
- Validate with healthcare professionals

### Accuracy
- Double-check medical terminology
- Verify drug dosages and protocols
- Include appropriate uncertainty indicators
- Add medical disclaimers for clinical features

### Scope
- Focus on emergency medicine use cases
- Consider critical care applications
- Include pre-hospital care scenarios
- Support clinical decision tools

## 🚀 Release Process

1. **Feature freeze** for release candidate
2. **Comprehensive testing** including medical accuracy
3. **Documentation review** and updates
4. **Security audit** for healthcare data handling
5. **Tag release** with semantic versioning
6. **Deploy to production** environments

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Medical Review**: Contact maintainers for clinical validation
- **Security Issues**: Use private reporting for security concerns

## 🎯 Project Priorities

1. **Medical Accuracy**: Ensuring clinical information is correct
2. **Performance**: Maintaining fast response times
3. **Usability**: Making the system accessible to healthcare workers
4. **Security**: Protecting sensitive medical information
5. **Extensibility**: Supporting additional medical domains

Thank you for contributing to advancing emergency medicine technology! 🚑