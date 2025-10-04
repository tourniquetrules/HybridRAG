#!/usr/bin/env python3
"""
Command-line interface for the Emergency Medicine Hybrid RAG System
"""

import argparse
import logging
import asyncio
import json
from pathlib import Path
import sys

from hybrid_rag import HybridRAGSystem
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_parser():
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Emergency Medicine Hybrid RAG System CLI"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents')
    ingest_parser.add_argument('path', help='Path to directory containing PDF documents')
    ingest_parser.add_argument('--reset', action='store_true', help='Reset existing database')
    ingest_parser.add_argument('--neo4j', action='store_true', help='Use Neo4j for knowledge graph')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the system')
    query_parser.add_argument('query', help='Query text')
    query_parser.add_argument('--no-vector', action='store_true', help='Disable vector search')
    query_parser.add_argument('--no-kg', action='store_true', help='Disable knowledge graph search')
    query_parser.add_argument('--no-rerank', action='store_true', help='Disable reranking')
    query_parser.add_argument('--max-results', type=int, default=10, help='Maximum results')
    query_parser.add_argument('--neo4j', action='store_true', help='Use Neo4j for knowledge graph')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Interactive chat mode')
    chat_parser.add_argument('--neo4j', action='store_true', help='Use Neo4j for knowledge graph')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    status_parser.add_argument('--neo4j', action='store_true', help='Use Neo4j for knowledge graph')
    
    # Web command
    web_parser = subparsers.add_parser('web', help='Start web server')
    web_parser.add_argument('--host', default=config.flask_host, help='Host to bind to')
    web_parser.add_argument('--port', type=int, default=config.flask_port, help='Port to bind to')
    web_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    return parser

def ingest_documents(args):
    """Ingest documents from directory"""
    try:
        logger.info(f"Initializing RAG system (Neo4j: {args.neo4j})")
        rag_system = HybridRAGSystem(use_neo4j=args.neo4j)
        
        path = Path(args.path)
        if not path.exists():
            logger.error(f"Path does not exist: {path}")
            return False
        
        if not path.is_dir():
            logger.error(f"Path is not a directory: {path}")
            return False
        
        pdf_files = list(path.glob("*.pdf"))
        if not pdf_files:
            logger.error(f"No PDF files found in: {path}")
            return False
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        success = rag_system.ingest_documents(str(path), reset_db=args.reset)
        
        if success:
            logger.info("Document ingestion completed successfully!")
            
            # Show system status
            status = rag_system.get_system_status()
            print("\n=== System Status ===")
            print(json.dumps(status, indent=2))
            
        else:
            logger.error("Document ingestion failed!")
            
        rag_system.close()
        return success
        
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        return False

def query_system(args):
    """Query the RAG system"""
    try:
        logger.info(f"Initializing RAG system (Neo4j: {args.neo4j})")
        rag_system = HybridRAGSystem(use_neo4j=args.neo4j)
        
        search_kwargs = {
            'use_vector': not args.no_vector,
            'use_knowledge_graph': not args.no_kg,
            'use_reranking': not args.no_rerank,
            'max_results': args.max_results
        }
        
        logger.info(f"Searching with settings: {search_kwargs}")
        
        # Perform search
        results = rag_system.search(args.query, **search_kwargs)
        
        print(f"\n=== Search Results for: '{args.query}' ===")
        print(f"Found {len(results)} results")
        
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(f"Text: {result.get('text', '')[:200]}...")
            print(f"Source: {result.get('metadata', {}).get('source', 'Unknown')}")
            print(f"Similarity Score: {result.get('similarity_score', 0):.3f}")
            if 'rerank_score' in result:
                print(f"Rerank Score: {result['rerank_score']:.3f}")
            print(f"Search Method: {result.get('search_method', 'unknown')}")
        
        rag_system.close()
        return True
        
    except Exception as e:
        logger.error(f"Error during query: {str(e)}")
        return False

def chat_mode(args):
    """Interactive chat mode"""
    try:
        logger.info(f"Initializing RAG system (Neo4j: {args.neo4j})")
        rag_system = HybridRAGSystem(use_neo4j=args.neo4j)
        
        print("\n" + "="*60)
        print("Emergency Medicine Hybrid RAG System - Chat Mode")
        print("="*60)
        print("Type 'exit' or 'quit' to end the session")
        print("Type 'status' to show system status")
        print("Type 'help' for more commands")
        print("="*60 + "\n")
        
        conversation_history = []
        
        while True:
            try:
                query = input("🩺 Ask me about emergency medicine: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['exit', 'quit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if query.lower() == 'status':
                    status = rag_system.get_system_status()
                    print("\n=== System Status ===")
                    print(json.dumps(status, indent=2))
                    continue
                
                if query.lower() == 'help':
                    print("\nAvailable commands:")
                    print("- status: Show system status")
                    print("- exit/quit: Exit chat mode")
                    print("- help: Show this help")
                    print("- clear: Clear conversation history")
                    continue
                
                if query.lower() == 'clear':
                    conversation_history = []
                    print("🗑️ Conversation history cleared")
                    continue
                
                print("🔍 Processing your query...")
                
                # Generate response
                response_data = rag_system.chat(query, conversation_history)
                
                print(f"\n🤖 Assistant: {response_data['response']}")
                
                # Show sources if available
                if response_data.get('sources'):
                    print(f"\n📚 Sources ({len(response_data['sources'])}):")
                    for i, source in enumerate(response_data['sources'][:3]):
                        print(f"  {i+1}. {source['source']} (Score: {source.get('similarity_score', 0):.3f})")
                
                # Update conversation history
                conversation_history.append({'role': 'user', 'content': query})
                conversation_history.append({'role': 'assistant', 'content': response_data['response']})
                
                # Keep only last 10 exchanges
                conversation_history = conversation_history[-20:]
                
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        rag_system.close()
        return True
        
    except Exception as e:
        logger.error(f"Error in chat mode: {str(e)}")
        return False

def show_status(args):
    """Show system status"""
    try:
        logger.info(f"Initializing RAG system (Neo4j: {args.neo4j})")
        rag_system = HybridRAGSystem(use_neo4j=args.neo4j)
        
        status = rag_system.get_system_status()
        
        print("\n=== Emergency Medicine Hybrid RAG System Status ===")
        print(json.dumps(status, indent=2))
        
        rag_system.close()
        return True
        
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return False

def start_web_server(args):
    """Start the web server"""
    try:
        # Import here to avoid circular imports
        from app import app
        
        logger.info(f"Starting web server on {args.host}:{args.port}")
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
        
    except Exception as e:
        logger.error(f"Error starting web server: {str(e)}")
        return False

def main():
    """Main CLI function"""
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Command dispatch
    commands = {
        'ingest': ingest_documents,
        'query': query_system,
        'chat': chat_mode,
        'status': show_status,
        'web': start_web_server
    }
    
    if args.command in commands:
        try:
            success = commands[args.command](args)
            return 0 if success else 1
        except KeyboardInterrupt:
            logger.info("Operation cancelled by user")
            return 1
    else:
        logger.error(f"Unknown command: {args.command}")
        parser.print_help()
        return 1

if __name__ == '__main__':
    sys.exit(main())