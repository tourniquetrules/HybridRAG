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

# Pydantic models
class ChatQuery(BaseModel):
    query: str
    use_vector: bool = True
    use_knowledge_graph: bool = True
    use_reranking: bool = True
    max_results: int = 10

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
    """Process chat query"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        logger.info(f"🔍 Processing query: {query.query[:100]}...")
        
        # Run search in executor to avoid blocking
        loop = asyncio.get_event_loop()
        search_results = await loop.run_in_executor(
            None, 
            rag_system.search,
            query.query,
            query.use_vector,
            query.use_knowledge_graph,
            query.use_reranking,
            query.max_results
        )
        
        if not search_results:
            return ChatResponse(
                response="I couldn't find any relevant information for your query. Please try rephrasing your question or check if documents have been uploaded.",
                sources=[],
                metadata={"query": query.query, "results_found": 0}
            )
        
        # Generate response using LLM
        chat_result = await loop.run_in_executor(
            None,
            rag_system.chat,
            query.query,
            None,  # conversation_history
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
        
        # Prepare sources
        sources = []
        for i, result in enumerate(search_results[:5]):  # Limit to top 5 sources
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
                "results_found": len(search_results),
                "processing_time": "async"
            }
        )
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

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