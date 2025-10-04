#!/bin/bash

# Setup script for Emergency Medicine Hybrid RAG System

echo "🏥 Setting up Emergency Medicine Hybrid RAG System..."

# Check if Python 3.8+ is available
python_cmd=""
if command -v python3.8 &> /dev/null; then
    python_cmd="python3.8"
elif command -v python3.9 &> /dev/null; then
    python_cmd="python3.9"
elif command -v python3.10 &> /dev/null; then
    python_cmd="python3.10"
elif command -v python3.11 &> /dev/null; then
    python_cmd="python3.11"
elif command -v python3 &> /dev/null; then
    python_cmd="python3"
elif command -v python &> /dev/null; then
    python_cmd="python"
else
    echo "❌ Python 3.8+ is required but not found"
    exit 1
fi

echo "✅ Using Python: $python_cmd"

# Create virtual environment
echo "📦 Creating virtual environment..."
$python_cmd -m venv venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python requirements..."
pip install -r requirements.txt

# Download spaCy model
echo "🧠 Downloading spaCy language model..."
python -m spacy download en_core_web_sm

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
    response = requests.get('http://192.168.2.64:1234/v1/models', timeout=5)
    if response.status_code == 200:
        print('✅ LM Studio is accessible')
    else:
        print('⚠️ LM Studio responded with status:', response.status_code)
except Exception as e:
    print('⚠️ Could not connect to LM Studio:', str(e))
    print('   Make sure LM Studio is running at http://192.168.2.64:1234')
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