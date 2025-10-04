#!/usr/bin/env python3
"""
HybridRAG Emergency Medicine Knowledge System
Setup configuration for package distribution
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
def read_requirements(filename):
    """Read requirements from file, filtering out comments and blank lines."""
    requirements = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-'):
                    # Handle version specifiers
                    if '>=' in line or '==' in line or '~=' in line:
                        requirements.append(line)
                    else:
                        requirements.append(line)
    return requirements

# Core requirements
install_requires = [
    'docling>=1.0.0',
    'chromadb>=0.4.0',
    'sentence-transformers>=2.2.0',
    'transformers>=4.30.0',
    'fastapi>=0.104.0',
    'uvicorn[standard]>=0.24.0',
    'jinja2>=3.1.0',
    'python-multipart>=0.0.6',
    'networkx>=3.0',
    'spacy>=3.7.0',
    'torch>=2.0.0',
    'numpy>=1.24.0',
    'pandas>=2.0.0',
    'scikit-learn>=1.3.0',
    'PyPDF2>=3.0.0',
    'pdf2image>=1.16.0',
    'pytesseract>=0.3.10',
    'requests>=2.31.0',
    'python-dotenv>=1.0.0',
    'pydantic>=2.0.0',
    'click>=8.1.0',
    'tqdm>=4.66.0',
]

# Optional dependencies for different use cases
extras_require = {
    'neo4j': ['neo4j>=5.0.0', 'py2neo>=2021.2.0'],
    'dev': [
        'pytest>=7.4.0',
        'pytest-asyncio>=0.21.0',
        'pytest-cov>=4.1.0',
        'black>=23.0.0',
        'flake8>=6.0.0',
        'mypy>=1.5.0',
        'bandit>=1.7.5',
        'safety>=2.3.0',
    ],
    'monitoring': [
        'prometheus-client>=0.17.0',
        'structlog>=23.1.0',
        'loguru>=0.7.0',
    ],
    'security': [
        'cryptography>=41.0.0',
        'python-jose>=3.3.0',
        'passlib>=1.7.4',
    ],
    'medical': [
        # Additional medical NLP packages
        # Note: spaCy models need to be installed separately
    ],
}

# All extras combined
extras_require['all'] = list(set(
    dep for deps in extras_require.values() for dep in deps
))

setup(
    name="hybridrag",
    version="1.0.0",
    author="HybridRAG Development Team",
    author_email="contact@hybridrag.com",
    description="Emergency Medicine Hybrid RAG System for Enhanced Medical Information Retrieval",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tourniquetrules/HybridRAG",
    project_urls={
        "Bug Tracker": "https://github.com/tourniquetrules/HybridRAG/issues",
        "Documentation": "https://github.com/tourniquetrules/HybridRAG/blob/main/README.md",
        "Source Code": "https://github.com/tourniquetrules/HybridRAG",
        "Changelog": "https://github.com/tourniquetrules/HybridRAG/blob/main/CHANGELOG.md",
    },
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    classifiers=[
        # Development Status
        "Development Status :: 5 - Production/Stable",
        
        # Intended Audience
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        
        # Topic
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
        "Topic :: Software Development :: Libraries :: Python Modules",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Programming Language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        
        # Operating System
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        
        # Framework
        "Framework :: FastAPI",
    ],
    keywords=[
        "rag", "retrieval-augmented-generation", "emergency-medicine", 
        "medical-ai", "knowledge-graph", "vector-search", "healthcare",
        "clinical-decision-support", "medical-nlp", "fastapi", "chromadb"
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "hybridrag=cli:main",
            "hybridrag-server=fastapi_app:main",
            "hybridrag-setup=setup:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": [
            "*.md", "*.txt", "*.yml", "*.yaml", "*.json",
            "templates/*", "static/*", "config/*"
        ],
    },
    zip_safe=False,
    
    # Medical and Safety Information
    license="MIT",
    platforms=["any"],
    
    # Additional metadata for healthcare applications
    project_metadata={
        "medical_disclaimer": "This software is for educational and research purposes only. "
                            "Not intended for direct clinical use without proper validation.",
        "specialty_focus": "Emergency Medicine",
        "data_privacy": "Designed for local deployment with no external data transmission",
        "compliance_notes": "Users responsible for HIPAA and other healthcare compliance requirements",
    },
)