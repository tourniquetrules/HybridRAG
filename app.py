from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import logging
import os
from datetime import datetime
import json
from typing import List, Dict, Any
import uuid
from werkzeug.utils import secure_filename

from hybrid_rag import HybridRAGSystem
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
CORS(app)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'PDF'}

# Initialize RAG system
rag_system = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def get_rag_system():
    """Get or initialize RAG system"""
    global rag_system
    if rag_system is None:
        try:
            rag_system = HybridRAGSystem(use_neo4j=False)  # Use NetworkX by default
            logger.info("RAG system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {str(e)}")
            return None
    return rag_system

@app.route('/')
def index():
    """Main chat interface"""
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Get or create session ID
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        # Get conversation history from session
        conversation_key = f"conversation_{session['session_id']}"
        conversation_history = session.get(conversation_key, [])
        
        # Get RAG system
        rag = get_rag_system()
        if not rag:
            return jsonify({'error': 'RAG system not available'}), 500
        
        # Get search parameters
        search_kwargs = {
            'use_vector': data.get('use_vector', True),
            'use_knowledge_graph': data.get('use_knowledge_graph', True),
            'use_reranking': data.get('use_reranking', True),
            'max_results': data.get('max_results', 10)
        }
        
        # Generate response
        response_data = rag.chat(query, conversation_history, search_kwargs)
        
        # Update conversation history
        conversation_history.append({'role': 'user', 'content': query})
        if response_data.get('response'):
            conversation_history.append({'role': 'assistant', 'content': response_data['response']})
        
        # Keep only last 20 messages
        conversation_history = conversation_history[-20:]
        session[conversation_key] = conversation_history
        
        # Add timestamp and session info
        response_data['timestamp'] = datetime.now().isoformat()
        response_data['session_id'] = session['session_id']
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/search', methods=['POST'])
def search():
    """Direct search endpoint"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        rag = get_rag_system()
        if not rag:
            return jsonify({'error': 'RAG system not available'}), 500
        
        # Search parameters
        search_kwargs = {
            'use_vector': data.get('use_vector', True),
            'use_knowledge_graph': data.get('use_knowledge_graph', True),
            'use_reranking': data.get('use_reranking', True),
            'max_results': data.get('max_results', 10)
        }
        
        # Perform search
        results = rag.search(query, **search_kwargs)
        
        return jsonify({
            'query': query,
            'results': results,
            'num_results': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models from LM Studio"""
    try:
        rag = get_rag_system()
        if not rag:
            return jsonify({'error': 'RAG system not available'}), 500
        
        models = rag.llm_client.get_available_models()
        current_model = rag.llm_client.get_current_model()
        
        return jsonify({
            'available_models': models,
            'current_model': current_model,
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        return jsonify({'error': f'Could not retrieve models: {str(e)}'}), 500

@app.route('/api/models/switch', methods=['POST'])
def switch_model():
    """Switch to a different model"""
    try:
        data = request.get_json()
        model_name = data.get('model_name', '').strip()
        
        if not model_name:
            return jsonify({'error': 'Model name is required'}), 400
        
        rag = get_rag_system()
        if not rag:
            return jsonify({'error': 'RAG system not available'}), 500
        
        success = rag.llm_client.switch_model(model_name)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Switched to model: {model_name}',
                'current_model': rag.llm_client.get_current_model(),
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Failed to switch to model: {model_name}',
                'timestamp': datetime.now().isoformat()
            }), 400
        
    except Exception as e:
        logger.error(f"Error switching model: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/ingest', methods=['POST'])
def ingest_documents():
    """Document ingestion endpoint"""
    try:
        data = request.get_json()
        documents_path = data.get('documents_path', '')
        reset_db = data.get('reset_db', False)
        
        if not documents_path:
            return jsonify({'error': 'documents_path is required'}), 400
        
        if not os.path.exists(documents_path):
            return jsonify({'error': 'Documents path does not exist'}), 400
        
        rag = get_rag_system()
        if not rag:
            return jsonify({'error': 'RAG system not available'}), 500
        
        # Start ingestion
        logger.info(f"Starting document ingestion from: {documents_path}")
        success = rag.ingest_documents(documents_path, reset_db)
        
        if success:
            return jsonify({
                'message': 'Document ingestion completed successfully',
                'documents_path': documents_path,
                'reset_db': reset_db,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Document ingestion failed'}), 500
            
    except Exception as e:
        logger.error(f"Error in ingestion endpoint: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """File upload endpoint for drag and drop"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        reset_db = request.form.get('reset_db', 'false').lower() == 'true'
        
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400
        
        rag = get_rag_system()
        if not rag:
            return jsonify({'error': 'RAG system not available'}), 500
        
        uploaded_files = []
        errors = []
        
        # Process each uploaded file
        for file in files:
            if file and file.filename != '':
                if not allowed_file(file.filename):
                    errors.append(f"File type not allowed: {file.filename}")
                    continue
                
                # Save file securely
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
                unique_filename = f"{timestamp}{filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                
                try:
                    file.save(file_path)
                    uploaded_files.append({
                        'original_name': file.filename,
                        'saved_name': unique_filename,
                        'path': file_path
                    })
                    logger.info(f"Saved uploaded file: {file_path}")
                except Exception as e:
                    errors.append(f"Failed to save {file.filename}: {str(e)}")
        
        if not uploaded_files:
            return jsonify({'error': 'No valid files uploaded', 'details': errors}), 400
        
        # Process uploaded files
        try:
            success_count = 0
            for file_info in uploaded_files:
                try:
                    logger.info(f"Processing uploaded file: {file_info['path']}")
                    # Process single file through the document processor
                    if rag.ingest_single_document(file_info['path']):
                        success_count += 1
                        logger.info(f"Successfully processed: {file_info['original_name']}")
                    else:
                        errors.append(f"Failed to process: {file_info['original_name']}")
                except Exception as e:
                    errors.append(f"Processing error for {file_info['original_name']}: {str(e)}")
            
            # Clean up uploaded files
            for file_info in uploaded_files:
                try:
                    if os.path.exists(file_info['path']):
                        os.remove(file_info['path'])
                except Exception as e:
                    logger.warning(f"Failed to clean up file {file_info['path']}: {str(e)}")
            
            response_data = {
                'message': f'File upload completed. Successfully processed {success_count} of {len(uploaded_files)} files.',
                'success_count': success_count,
                'total_files': len(uploaded_files),
                'processed_files': [f['original_name'] for f in uploaded_files],
                'timestamp': datetime.now().isoformat()
            }
            
            if errors:
                response_data['errors'] = errors
                response_data['warning'] = 'Some files had issues'
            
            return jsonify(response_data)
            
        except Exception as e:
            # Clean up files on processing error
            for file_info in uploaded_files:
                try:
                    if os.path.exists(file_info['path']):
                        os.remove(file_info['path'])
                except:
                    pass
            raise e
            
    except Exception as e:
        logger.error(f"Error in upload endpoint: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """System status endpoint"""
    try:
        rag = get_rag_system()
        if not rag:
            return jsonify({'error': 'RAG system not available'}), 500
        
        status = rag.get_system_status()
        status['timestamp'] = datetime.now().isoformat()
        status['config'] = {
            'embedding_model': config.embedding_model,
            'reranker_model': config.reranker_model,
            'lm_studio_host': config.lm_studio_host,
            'vector_db_path': config.vector_db_path
        }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error in status endpoint: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    try:
        if 'session_id' in session:
            conversation_key = f"conversation_{session['session_id']}"
            session.pop(conversation_key, None)
        
        return jsonify({
            'message': 'Conversation history cleared',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/export_conversation', methods=['GET'])
def export_conversation():
    """Export conversation history"""
    try:
        if 'session_id' not in session:
            return jsonify({'error': 'No active session'}), 400
        
        conversation_key = f"conversation_{session['session_id']}"
        conversation_history = session.get(conversation_key, [])
        
        return jsonify({
            'session_id': session['session_id'],
            'conversation': conversation_history,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error exporting conversation: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting Emergency Medicine Hybrid RAG System")
    logger.info(f"Config: {config.flask_host}:{config.flask_port}")
    
    app.run(
        host=config.flask_host,
        port=config.flask_port,
        debug=config.flask_debug
    )