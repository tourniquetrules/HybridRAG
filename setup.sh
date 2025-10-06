#!/bin/bash

# Setup script for Emergency Medicine Hybrid RAG System

echo "🏥 Setting up Emergency Medicine Hybrid RAG System..."

# Check if Python 3.10.x is available (required for spaCy SciBERT)
python_cmd=""
if command -v python3.10 &> /dev/null; then
    python_cmd="python3.10"
elif command -v python3 &> /dev/null; then
    # Check if python3 is actually 3.10
    python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -n1)
    if [[ "$python_version" == "3.10" ]]; then
        python_cmd="python3"
    else
        echo "❌ Python 3.10.x is required for spaCy SciBERT medical models"
        echo "   Found Python $python_version, but need 3.10.x"
        echo "   Please install Python 3.10.x and try again"
        exit 1
    fi
elif command -v python &> /dev/null; then
    # Check if python is actually 3.10
    python_version=$(python --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -n1)
    if [[ "$python_version" == "3.10" ]]; then
        python_cmd="python"
    else
        echo "❌ Python 3.10.x is required for spaCy SciBERT medical models"
        echo "   Found Python $python_version, but need 3.10.x"
        echo "   Please install Python 3.10.x and try again"
        exit 1
    fi
else
    echo "❌ Python 3.10.x is required but not found"
    echo "   Please install Python 3.10.x and try again"
    exit 1
fi

echo "✅ Using Python: $python_cmd"

# Create virtual environment with Python 3.10
echo "📦 Creating Python 3.10 virtual environment..."
$python_cmd -m venv venv_py310

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    source venv_py310/Scripts/activate
else
    source venv_py310/bin/activate
fi

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python requirements..."
pip install -r requirements.txt

# Download spaCy models
echo "🧠 Downloading spaCy language models..."
python -m spacy download en_core_web_sm

# Install spaCy SciBERT for medical entity extraction
echo "🏥 Installing spaCy SciBERT medical model..."
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz

# Create directories
echo "📁 Creating directories..."
mkdir -p vector_db
mkdir -p documents
mkdir -p logs

# Set permissions for CLI
chmod +x cli.py

# Check LM Studio connection
echo "🔌 Checking LM Studio connection..."
python -c "
import requests
try:
    response = requests.get('http://192.168.2.180:1234/v1/models', timeout=5)
    if response.status_code == 200:
        print('✅ LM Studio is accessible')
    else:
        print('⚠️ LM Studio responded with status:', response.status_code)
except Exception as e:
    print('⚠️ Could not connect to LM Studio:', str(e))
    print('   Make sure LM Studio is running at http://192.168.2.180:1234')
"

# Check if Neo4j is available (optional)
echo "🔍 Checking Neo4j availability (optional)..."
python -c "
try:
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
    with driver.session() as session:
        result = session.run('RETURN 1')
        print('✅ Neo4j is available and accessible')
    driver.close()
except Exception as e:
    print('⚠️ Neo4j not available (will use NetworkX instead):', str(e))
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Place your emergency medicine PDF documents in the 'documents' folder"
echo "2. Activate the virtual environment: source venv/bin/activate"
echo "3. Ingest documents: python cli.py ingest documents/"
echo "4. Start the web interface: python cli.py web"
echo "   OR use interactive chat: python cli.py chat"
echo ""
echo "Web interface will be available at: http://localhost:5000"
echo "API endpoints:"
echo "  - POST /api/chat - Chat with the system"
echo "  - POST /api/search - Direct search"
echo "  - POST /api/ingest - Ingest documents"
echo "  - GET /api/status - System status"
echo ""
echo "For help: python cli.py --help"