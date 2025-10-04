#!/bin/bash
# Enhanced activation script for Emergency Medicine RAG System with Python 3.10
# Optimized for larger models and scientific computing

echo "🏥 Activating Emergency Medicine Hybrid RAG System (Python 3.10)"
echo "==============================================================="

# Activate Python 3.10 virtual environment
source venv_py310/bin/activate

# Verify Python version
PYTHON_VERSION=$(python --version)
echo "🐍 Python Version: $PYTHON_VERSION"

if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual Environment: $(basename $VIRTUAL_ENV)"
else
    echo "⚠️  Warning: Not in virtual environment"
    exit 1
fi

# Set memory and GPU optimizations for larger models
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="/tmp/hf_cache"

echo "🚀 Environment Variables Set:"
echo "   - PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "   - CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "   - TOKENIZERS_PARALLELISM: $TOKENIZERS_PARALLELISM"

# Check if basic dependencies exist
echo "📦 Checking Dependencies..."
if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
    CUDA_AVAILABLE=$(python -c "import torch; print('Yes' if torch.cuda.is_available() else 'No')" 2>/dev/null)
    echo "   ✅ PyTorch: $TORCH_VERSION (CUDA: $CUDA_AVAILABLE)"
else
    echo "   📥 PyTorch not installed - run: pip install torch torchvision torchaudio"
fi

if python -c "import transformers" 2>/dev/null; then
    TRANSFORMERS_VERSION=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null)
    echo "   ✅ Transformers: $TRANSFORMERS_VERSION"
else
    echo "   📥 Transformers not installed - run: pip install transformers"
fi

if python -c "import sentence_transformers" 2>/dev/null; then
    ST_VERSION=$(python -c "import sentence_transformers; print(sentence_transformers.__version__)" 2>/dev/null)
    echo "   ✅ Sentence Transformers: $ST_VERSION"
else
    echo "   📥 Sentence Transformers not installed"
fi

# Check GPU memory
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | head -1 | while read gpu_info; do
        echo "   GPU: $gpu_info MB"
    done
fi

# Install core dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📋 Requirements file found"
    read -p "Install/upgrade dependencies from requirements.txt? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📦 Installing dependencies..."
        pip install -r requirements.txt
    fi
else
    echo "📋 Creating requirements.txt for Python 3.10 environment..."
fi

# Check system status
echo ""
echo "📊 System Status:"
echo "- Vector DB: $(ls -lah vector_db/chroma.sqlite3 2>/dev/null | awk '{print $5}' || echo 'Not found')"
echo "- Knowledge Graph: $([ -f vector_db/knowledge_graph.pkl ] && echo 'Available' || echo 'Will be created on first run')"
echo "- Upload Directory: $([ -d uploads ] && echo 'Ready' || echo 'Will be created')"
echo "- Python Environment: $(basename $VIRTUAL_ENV)"

echo ""
echo "🔬 Ready for Scientific Computing & Large Models!"
echo "🌐 To start the system: python app.py"
echo "📚 Web Interface: http://localhost:5000"
echo "💡 For larger models, consider installing: transformers[torch], accelerate, bitsandbytes"