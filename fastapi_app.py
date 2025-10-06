from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import asyncio
import aiofiles
import os
import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid
from pathlib import Path
import traceback

from hybrid_rag import HybridRAGSystem
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Emergency Medicine RAG System",
    description="Advanced RAG system with semantic chunking and medical intelligence",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Templates and static files
templates = Jinja2Templates(directory="templates")
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Ensure upload folder exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'pdf', 'docx', 'html', 'md', 'csv', 'xlsx', 'pptx', 'txt'
}

# Global RAG system instance
rag_system: Optional[HybridRAGSystem] = None

# In-memory session storage for conversation history
conversation_sessions: Dict[str, List[Dict[str, str]]] = {}

# Pydantic models
class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatQuery(BaseModel):
    query: str
    use_vector: bool = True
    use_knowledge_graph: bool = True
    use_reranking: bool = True
    max_results: int = 10
    conversation_history: List[ChatMessage] = []
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class SystemStatus(BaseModel):
    status: str
    rag_initialized: bool
    document_count: int
    knowledge_graph_nodes: int
    knowledge_graph_relationships: int
    available_models: int

class GPUEndpoint(BaseModel):
    endpoint: str

class GPUResponse(BaseModel):
    success: bool
    message: str
    
class GPUConnectionTest(BaseModel):
    connected: bool
    message: str

class BenchmarkRequest(BaseModel):
    concurrent_users: int = 10
    queries_per_user: int = 5
    test_query: str = "What is the management of septic shock?"

class BenchmarkResult(BaseModel):
    success: bool
    message: str
    metrics: Dict[str, Any] = {}

class ModelSwitchRequest(BaseModel):
    model_name: str

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    global rag_system
    try:
        logger.info("🚀 Starting Enhanced Emergency Medicine RAG System")
        logger.info(f"📍 Server running on: {config.flask_host}:{config.flask_port}")
        
        # Initialize RAG system in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        rag_system = await loop.run_in_executor(None, HybridRAGSystem)
        
        logger.info("✅ RAG system initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {str(e)}")
        rag_system = None

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🔄 Shutting down Enhanced RAG System...")
    # Add any cleanup logic here if needed

# Helper functions
def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

async def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file and return the path"""
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail=f"File type not allowed: {file.filename}")
    
    # Create unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    # Save file asynchronously
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    logger.info(f"📁 Saved uploaded file: {file_path}")
    return file_path

async def process_document_async(file_path: str) -> Dict[str, Any]:
    """Process a single document asynchronously"""
    try:
        logger.info(f"🔄 Processing document: {file_path}")
        
        # Run document processing in executor to avoid blocking
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None, rag_system.ingest_single_document, file_path
        )
        
        if success:
            logger.info(f"✅ Successfully processed: {file_path}")
            return {"status": "success", "file": file_path}
        else:
            logger.error(f"❌ Failed to process: {file_path}")
            return {"status": "error", "file": file_path, "error": "Processing failed"}
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Error processing {file_path}: {error_msg}")
        return {"status": "error", "file": file_path, "error": error_msg}

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main chat interface"""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/api/status")
async def get_status():
    """Get system status"""
    try:
        if not rag_system:
            return {
                "status": "initializing",
                "rag_initialized": False,
                "vector_database": {
                    "status": "initializing",
                    "total_documents": 0
                },
                "knowledge_graph": {
                    "status": "initializing",
                    "type": "NetworkX",
                    "nodes": 0,
                    "relationships": 0
                },
                "llm": {
                    "status": "initializing",
                    "available_models": 0
                }
            }
        
        # Get document count
        doc_count = rag_system.vector_db.collection.count() if rag_system.vector_db else 0
        
        # Get knowledge graph stats
        kg_nodes = len(rag_system.knowledge_graph.graph.nodes()) if rag_system.knowledge_graph else 0
        kg_edges = len(rag_system.knowledge_graph.graph.edges()) if rag_system.knowledge_graph else 0
        
        # Get available models
        model_info = rag_system.llm_client.get_model_info() if rag_system.llm_client else {}
        available_models = len(model_info.get('data', [])) if model_info else 0
        
        return {
            "status": "ready",
            "rag_initialized": True,
            "vector_database": {
                "status": "ready",
                "total_documents": doc_count
            },
            "knowledge_graph": {
                "status": "ready",
                "type": "NetworkX",
                "nodes": kg_nodes,
                "relationships": kg_edges
            },
            "llm": {
                "status": "ready",
                "available_models": available_models
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return {
            "status": "error",
            "rag_initialized": False,
            "vector_database": {
                "status": "error",
                "total_documents": 0
            },
            "knowledge_graph": {
                "status": "error",
                "type": "NetworkX",
                "nodes": 0,
                "relationships": 0
            },
            "llm": {
                "status": "error",
                "available_models": 0
            }
        }

@app.get("/api/models")
async def get_models():
    """Get available models from LM Studio"""
    try:
        if not rag_system:
            raise HTTPException(status_code=500, detail="RAG system not initialized")
        
        models = rag_system.llm_client.get_available_models()
        current_model = rag_system.llm_client.get_current_model()
        
        return {
            "available_models": models,
            "current_model": current_model,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Could not retrieve models: {str(e)}")

@app.post("/api/models/switch")
async def switch_model(request: ModelSwitchRequest):
    """Switch to a different model"""
    try:
        if not rag_system:
            raise HTTPException(status_code=500, detail="RAG system not initialized")
        
        if not request.model_name.strip():
            raise HTTPException(status_code=400, detail="Model name is required")
        
        success = rag_system.llm_client.switch_model(request.model_name)
        
        if success:
            return {
                "status": "success",
                "message": f"Switched to model: {request.model_name}",
                "current_model": rag_system.llm_client.get_current_model(),
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to switch to model: {request.model_name}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/documents/summary")
async def get_document_processing_summary():
    """Get comprehensive document processing summary"""
    try:
        if not rag_system:
            raise HTTPException(status_code=503, detail="RAG system not initialized")
        
        summary = rag_system.get_document_processing_summary()
        return summary
        
    except Exception as e:
        logger.error(f"Error getting document summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting document summary: {str(e)}")

@app.post("/api/switch_gpu", response_model=GPUResponse)
async def switch_gpu_endpoint(gpu_endpoint: GPUEndpoint):
    """Switch to a different GPU endpoint"""
    try:
        from llm_client import LMStudioClient
        
        # Update the global configuration
        config.lm_studio_host = gpu_endpoint.endpoint
        os.environ['LM_STUDIO_HOST'] = gpu_endpoint.endpoint
        
        # Reinitialize the LLM client with new endpoint
        global rag_system
        if rag_system and hasattr(rag_system, 'llm_client'):
            rag_system.llm_client = LMStudioClient()
            
        return GPUResponse(
            success=True,
            message=f"Successfully switched to endpoint: {gpu_endpoint.endpoint}"
        )
    except Exception as e:
        logger.error(f"Error switching GPU endpoint: {e}")
        return GPUResponse(
            success=False,
            message=f"Failed to switch endpoint: {str(e)}"
        )

@app.post("/api/test_gpu_connection", response_model=GPUConnectionTest)
async def test_gpu_connection(gpu_endpoint: GPUEndpoint):
    """Test connection to a GPU endpoint"""
    import requests
    
    try:
        # Test connection with a simple HTTP request
        test_url = f"{gpu_endpoint.endpoint}/v1/models"
        response = requests.get(test_url, timeout=5)
        
        if response.status_code == 200:
            models_data = response.json()
            model_count = len(models_data.get('data', []))
            return GPUConnectionTest(
                connected=True,
                message=f"Connected - {model_count} model(s) available"
            )
        else:
            return GPUConnectionTest(
                connected=False,
                message=f"HTTP {response.status_code}: {response.text[:100]}"
            )
            
    except requests.exceptions.Timeout:
        return GPUConnectionTest(
            connected=False,
            message="Connection timeout"
        )
    except requests.exceptions.ConnectionError:
        return GPUConnectionTest(
            connected=False,
            message="Connection refused"
        )
    except Exception as e:
        return GPUConnectionTest(
            connected=False,
            message=f"Connection failed: {str(e)}"
        )

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and process multiple files"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not files or files[0].filename == '':
        raise HTTPException(status_code=400, detail="No files selected")
    
    results = []
    
    try:
        # Save all files first
        file_paths = []
        for file in files:
            file_path = await save_uploaded_file(file)
            file_paths.append(file_path)
        
        # Process documents concurrently
        processing_tasks = [process_document_async(path) for path in file_paths]
        results = await asyncio.gather(*processing_tasks, return_exceptions=True)
        
        # Count successes and failures
        successful = sum(1 for r in results if isinstance(r, dict) and r.get('status') == 'success')
        failed = len(results) - successful
        
        return JSONResponse({
            "status": "completed",
            "message": f"Processed {len(files)} files: {successful} successful, {failed} failed",
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat(query: ChatQuery):
    """Process chat query with conversation context"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        logger.info(f"🔍 Processing query: {query.query[:100]}...")
        
        # Handle session management
        session_id = query.session_id or str(uuid.uuid4())
        
        # Get or initialize conversation history
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = []
        
        # Convert Pydantic models to dict format expected by the backend
        conversation_history = []
        if query.conversation_history:
            conversation_history = [
                {"role": msg.role, "content": msg.content} 
                for msg in query.conversation_history
            ]
        else:
            # Use session-stored history if no history provided in request
            conversation_history = conversation_sessions[session_id]
        
        # Generate response using LLM with conversation context
        loop = asyncio.get_event_loop()
        chat_result = await loop.run_in_executor(
            None,
            rag_system.chat,
            query.query,
            conversation_history,  # Pass conversation history
            {
                "use_vector": query.use_vector,
                "use_knowledge_graph": query.use_knowledge_graph,
                "use_reranking": query.use_reranking,
                "max_results": query.max_results
            }
        )
        
        response = chat_result.get('response', 'No response generated')
        # Update search_results from chat result if available
        if 'sources' in chat_result:
            search_results = chat_result['sources']
        
        # Update conversation history in session
        conversation_sessions[session_id].append({"role": "user", "content": query.query})
        conversation_sessions[session_id].append({"role": "assistant", "content": response})
        
        # Keep only last 20 messages to prevent memory bloat
        conversation_sessions[session_id] = conversation_sessions[session_id][-20:]
        
        # Prepare sources
        sources = []
        search_results_list = search_results if isinstance(search_results, list) else []
        for i, result in enumerate(search_results_list[:5]):  # Limit to top 5 sources
            if isinstance(result, dict):
                sources.append({
                    "index": i + 1,
                    "text": result.get('text', '')[:200] + "..." if len(result.get('text', '')) > 200 else result.get('text', ''),
                    "metadata": result.get('metadata', {}),
                    "score": result.get('score', 0)
                })
        
        return ChatResponse(
            response=response,
            sources=sources,
            metadata={
                "query": query.query,
                "results_found": len(search_results_list),
                "processing_time": "async",
                "session_id": session_id,
                "conversation_length": len(conversation_sessions[session_id])
            }
        )
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/api/chat/clear")
async def clear_conversation(session_id: str = None):
    """Clear conversation history for a session"""
    try:
        if session_id and session_id in conversation_sessions:
            del conversation_sessions[session_id]
            return {"status": "success", "message": f"Cleared conversation for session: {session_id}"}
        elif session_id:
            return {"status": "info", "message": "Session not found or already empty"}
        else:
            # Clear all sessions
            conversation_sessions.clear()
            return {"status": "success", "message": "Cleared all conversations"}
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/history")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    try:
        if session_id in conversation_sessions:
            return {
                "session_id": session_id,
                "conversation_history": conversation_sessions[session_id],
                "message_count": len(conversation_sessions[session_id])
            }
        else:
            return {
                "session_id": session_id,
                "conversation_history": [],
                "message_count": 0
            }
    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/stream")
async def stream_chat(query: str):
    """Stream chat responses (for future real-time implementation)"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    async def generate_stream():
        try:
            # For now, just return the regular response as a stream
            # This can be enhanced later for true streaming
            chat_query = ChatQuery(query=query)
            response = await chat(chat_query)
            
            # Simulate streaming by breaking response into chunks
            words = response.response.split()
            for i in range(0, len(words), 5):  # 5 words at a time
                chunk = " ".join(words[i:i+5]) + " "
                yield f"data: {json.dumps({'chunk': chunk, 'done': False})}\n\n"
                await asyncio.sleep(0.1)  # Small delay for streaming effect
            
            yield f"data: {json.dumps({'chunk': '', 'done': True, 'sources': response.sources})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.delete("/api/documents/clear")
async def clear_documents():
    """Clear all documents from the system"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Clear vector database
        await asyncio.get_event_loop().run_in_executor(
            None, rag_system.clear_database
        )
        
        return JSONResponse({
            "status": "success",
            "message": "All documents cleared successfully"
        })
        
    except Exception as e:
        logger.error(f"Clear documents error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found"}
        )

@app.post("/api/benchmark_vector_db", response_model=BenchmarkResult)
async def benchmark_vector_db(request: BenchmarkRequest):
    """Run vector database benchmark test"""
    import time
    import statistics
    import threading
    import psutil
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    try:
        if not rag_system or not rag_system.vector_db:
            return BenchmarkResult(
                success=False,
                message="RAG system not initialized",
                metrics={}
            )
        
        def single_query_test(query: str) -> dict:
            """Test a single query timing"""
            start_time = time.time()
            
            try:
                # Time embedding
                embedding_start = time.time()
                embedding = rag_system.vector_db.embedding_model.encode([query])
                embedding_time = time.time() - embedding_start
                
                # Time search
                search_start = time.time()
                results = rag_system.vector_db.collection.query(
                    query_embeddings=embedding,
                    n_results=5,
                    include=['documents', 'metadatas', 'distances']
                )
                search_time = time.time() - search_start
                
                total_time = time.time() - start_time
                
                return {
                    'embedding_time': embedding_time,
                    'search_time': search_time,
                    'total_time': total_time,
                    'success': True,
                    'results_count': len(results['documents'][0]) if results['documents'] else 0
                }
            except Exception as e:
                return {
                    'embedding_time': 0,
                    'search_time': 0,
                    'total_time': time.time() - start_time,
                    'success': False,
                    'error': str(e),
                    'results_count': 0
                }
        
        # Collect system metrics before test
        cpu_before = psutil.cpu_percent()
        mem_before = psutil.virtual_memory().percent
        
        # Run concurrent test
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=request.concurrent_users) as executor:
            futures = []
            for _ in range(request.concurrent_users * request.queries_per_user):
                future = executor.submit(single_query_test, request.test_query)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=10)
                    results.append(result)
                except Exception as e:
                    results.append({
                        'embedding_time': 0,
                        'search_time': 0,
                        'total_time': 0,
                        'success': False,
                        'error': str(e),
                        'results_count': 0
                    })
        
        total_duration = time.time() - start_time
        
        # Collect system metrics after test
        cpu_after = psutil.cpu_percent()
        mem_after = psutil.virtual_memory().percent
        
        # Calculate metrics
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        if successful_results:
            total_times = [r['total_time'] for r in successful_results]
            embedding_times = [r['embedding_time'] for r in successful_results]
            search_times = [r['search_time'] for r in successful_results]
            
            metrics = {
                'total_queries': len(results),
                'successful_queries': len(successful_results),
                'failed_queries': len(failed_results),
                'success_rate': len(successful_results) / len(results) * 100,
                'concurrent_users': request.concurrent_users,
                'queries_per_user': request.queries_per_user,
                'total_duration': total_duration,
                'queries_per_second': len(successful_results) / total_duration,
                'avg_total_time_ms': statistics.mean(total_times) * 1000,
                'avg_embedding_time_ms': statistics.mean(embedding_times) * 1000,
                'avg_search_time_ms': statistics.mean(search_times) * 1000,
                'min_time_ms': min(total_times) * 1000,
                'max_time_ms': max(total_times) * 1000,
                'p50_time_ms': statistics.median(total_times) * 1000,
                'p95_time_ms': sorted(total_times)[int(len(total_times) * 0.95)] * 1000,
                'cpu_usage_before': cpu_before,
                'cpu_usage_after': cpu_after,
                'memory_usage_before': mem_before,
                'memory_usage_after': mem_after,
                'errors': [r.get('error', '') for r in failed_results if r.get('error')]
            }
        else:
            metrics = {
                'total_queries': len(results),
                'successful_queries': 0,
                'failed_queries': len(failed_results),
                'success_rate': 0,
                'concurrent_users': request.concurrent_users,
                'queries_per_user': request.queries_per_user,
                'total_duration': total_duration,
                'queries_per_second': 0,
                'errors': [r.get('error', '') for r in failed_results if r.get('error')]
            }
        
        return BenchmarkResult(
            success=True,
            message=f"Benchmark completed: {len(successful_results)}/{len(results)} queries successful",
            metrics=metrics
        )
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return BenchmarkResult(
            success=False,
            message=f"Benchmark failed: {str(e)}",
            metrics={}
        )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Enhanced Emergency Medicine RAG System with FastAPI")
    logger.info(f"📍 Config: {config.flask_host}:{config.flask_port}")
    
    uvicorn.run(
        "fastapi_app:app",
        host=config.flask_host,
        port=config.flask_port,
        reload=True,
        log_level="info",
        access_log=True
    )