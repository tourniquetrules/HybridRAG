#!/bin/bash
# Activation script for Emergency Medicine RAG System

echo "🏥 Activating Emergency Medicine Hybrid RAG System Environment"
echo "============================================================="

# Activate virtual environment
source venv_docling/bin/activate

# Check if in virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual Environment: $(basename $VIRTUAL_ENV)"
else
    echo "⚠️  Warning: Not in virtual environment"
fi

# Show Python path
echo "🐍 Python Path: $(which python)"

# Check if dependencies are installed
if python -c "import docling, chromadb, flask" 2>/dev/null; then
    echo "✅ Core dependencies installed"
else
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Check system status
echo ""
echo "📊 System Status:"
echo "- Vector DB: $(ls -lah vector_db/chroma.sqlite3 2>/dev/null | awk '{print $5}' || echo 'Not found')"
echo "- Knowledge Graph: $([ -f vector_db/knowledge_graph.pkl ] && echo 'Available' || echo 'Will be created on first run')"
echo "- Upload Directory: $([ -d uploads ] && echo 'Ready' || echo 'Will be created')"

echo ""
echo "🌐 Ready to start! Run: python app.py"
echo "📚 Web Interface: http://localhost:5000"