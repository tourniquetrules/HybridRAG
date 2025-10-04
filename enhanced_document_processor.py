import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import spacy
from transformers import AutoTokenizer

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.document import ConversionResult
from docling.datamodel.base_models import InputFormat, ConversionStatus

from config import config

logger = logging.getLogger(__name__)

class EnhancedDocumentProcessor:
    """Enhanced document processor using Docling for multiple formats + spaCy SciBERT for semantic chunking"""
    
    def __init__(self):
        """Initialize the enhanced processor with multi-format support and semantic chunking"""
        
        # Configure PDF pipeline for high-quality medical document processing
        pdf_pipeline_options = PdfPipelineOptions()
        pdf_pipeline_options.do_ocr = True  # Enable OCR for scanned documents
        pdf_pipeline_options.do_table_structure = True  # Extract table structure
        pdf_pipeline_options.table_structure_options.do_cell_matching = True
        
        # Configure Docling converter for multiple input formats
        self.converter = DocumentConverter(
            format_options={
                # PDF with advanced options for medical documents
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
                # Add support for other formats (using defaults for now)
                InputFormat.HTML: {},  # Clinical guidelines from websites
                InputFormat.DOCX: {},  # Hospital policies and procedures  
                InputFormat.XLSX: {},  # Medical datasets and protocols
                InputFormat.CSV: {},   # Data files
                InputFormat.MD: {},    # Research notes and documentation
                InputFormat.PPTX: {},  # Medical presentations
                InputFormat.IMAGE: {} # Scanned documents, charts
            }
        )
        
        # Initialize spaCy SciBERT for medical semantic processing
        try:
            self.nlp = spacy.load("en_core_sci_scibert")
            logger.info("Loaded scientific spaCy model: en_core_sci_scibert")
            self.semantic_chunking_enabled = True
        except OSError:
            logger.warning("Scientific spaCy model not available, falling back to word-based chunking")
            self.nlp = None
            self.semantic_chunking_enabled = False
            
        # Initialize tokenizer for token counting
        self.tokenizer = AutoTokenizer.from_pretrained(config.embedding_model)
        
    def process_document(self, file_path: str) -> Optional[ConversionResult]:
        """
        Process a single document using Docling (supports multiple formats)
        
        Args:
            file_path: Path to the document file
            
        Returns:
            ConversionResult containing processed document or None if failed
        """
        try:
            logger.info(f"Processing document: {file_path}")
            
            # Docling automatically detects file format and uses appropriate converter
            result = self.converter.convert(file_path)
            
            # Check conversion status using Docling v2 API
            if result.status in [ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS]:
                logger.info(f"Successfully processed: {file_path}")
                return result
            else:
                logger.error(f"Failed to process document: {file_path}, status: {result.status}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {type(e).__name__}: {str(e)}")
            if hasattr(e, 'args') and e.args:
                logger.error(f"Exception details: {e.args}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def extract_chunks(self, conversion_result: ConversionResult) -> List[Dict[str, Any]]:
        """
        Extract chunks from processed document with semantic chunking support
        
        Args:
            conversion_result: Result from Docling conversion
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        try:
            document = conversion_result.document
            chunks = []
            
            # Extract main document text
            main_text = document.export_to_markdown()
            
            # Create metadata (ensure all values are ChromaDB-compatible types)
            metadata = {
                "source": str(conversion_result.input.file),  # Convert Path to string
                "format": str(conversion_result.input.format.value) if hasattr(conversion_result.input.format, 'value') else str(conversion_result.input.format),
                "title": getattr(document, 'title', Path(conversion_result.input.file).stem),
                "conversion_status": conversion_result.status.name
            }
            
            # Choose chunking strategy based on availability of semantic processing
            if self.semantic_chunking_enabled:
                text_chunks = self._semantic_split_text(
                    main_text, 
                    config.chunk_size, 
                    config.chunk_overlap
                )
                chunking_method = "semantic_scibert"
            else:
                text_chunks = self._split_text(
                    main_text, 
                    config.chunk_size, 
                    config.chunk_overlap
                )
                chunking_method = "word_based"
            
            # Create chunks with metadata (ensure all values are ChromaDB-compatible)
            for i, chunk_text in enumerate(text_chunks):
                chunk = {
                    "text": chunk_text,
                    "chunk_id": f"{Path(str(conversion_result.input.file)).stem}_chunk_{i}",
                    "metadata": {
                        **metadata,
                        "chunk_index": i,
                        "total_chunks": len(text_chunks),
                        "chunking_method": chunking_method,
                        "tokens": len(self.tokenizer.encode(chunk_text))
                    }
                }
                chunks.append(chunk)
                
            # Extract tables as separate chunks (Docling handles this for all formats)
            table_chunks = self._extract_table_chunks(document, metadata)
            chunks.extend(table_chunks)
            
            logger.info(f"Extracted {len(chunks)} chunks from document using {chunking_method} chunking")
            return chunks
            
        except Exception as e:
            logger.error(f"Error extracting chunks: {str(e)}")
            return []
    
    def _semantic_split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Split text using semantic boundaries with spaCy SciBERT
        
        Args:
            text: Input text to split
            chunk_size: Maximum tokens per chunk
            overlap: Token overlap between chunks
            
        Returns:
            List of semantically coherent text chunks
        """
        if not self.nlp:
            return self._split_text(text, chunk_size, overlap)
        
        # Process text with spaCy SciBERT
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            # Check if sentence starts a new section (medical papers often have clear sections)
            section_markers = ['Abstract:', 'Introduction:', 'Methods:', 'Results:', 'Discussion:', 'Conclusions:', 'Background:', 'Objective:', 'Design:', 'Setting:', 'Participants:', 'Interventions:', 'Outcomes:']
            is_new_section = any(sentence.strip().startswith(marker) for marker in section_markers)
            
            if is_new_section and current_chunk:
                # Save current chunk and start new section
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            elif current_tokens + sentence_tokens <= chunk_size:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            else:
                # Current chunk is full, start new chunk
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(chunk_text)
                
                # Handle overlap by including last few sentences
                overlap_sentences = []
                overlap_tokens = 0
                for sent in reversed(current_chunk):
                    sent_tokens = len(self.tokenizer.encode(sent))
                    if overlap_tokens + sent_tokens <= overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_tokens += sent_tokens
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_tokens = overlap_tokens + sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks
    
    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Fallback word-based splitting (original method)"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)
            
            if len(chunk_words) < chunk_size:
                break
                
        return chunks
    
    def _extract_table_chunks(self, document: Any, base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tables as separate chunks (works for all Docling-supported formats)"""
        table_chunks = []
        
        try:
            # Access tables from the DoclingDocument
            for i, table in enumerate(document.tables):
                # Convert table to markdown format
                table_text = self._table_to_markdown(table)
                
                if table_text.strip():
                    table_chunk = {
                        "text": table_text,
                        "chunk_id": f"{Path(str(base_metadata['source'])).stem}_table_{i}",
                        "metadata": {
                            **base_metadata,
                            "chunk_type": "table",
                            "table_index": i,
                            "chunking_method": "structured_table",
                            "tokens": len(self.tokenizer.encode(table_text))
                        }
                    }
                    table_chunks.append(table_chunk)
                    
        except Exception as e:
            logger.warning(f"Error extracting tables: {str(e)}")
            
        return table_chunks
    
    def _table_to_markdown(self, table: Any) -> str:
        """Convert table to markdown format"""
        try:
            # This is a simplified version - adjust based on your table structure needs
            if hasattr(table, 'export_to_markdown'):
                return table.export_to_markdown()
            else:
                return str(table)
        except Exception as e:
            logger.warning(f"Error converting table to markdown: {str(e)}")
            return ""

    def get_supported_formats(self) -> List[str]:
        """Get list of supported document formats"""
        return [
            'PDF', 'HTML', 'DOCX', 'XLSX', 'CSV', 
            'MD', 'PPTX', 'IMAGE', 'ASCIIDOC'
        ]
    
    def get_chunking_info(self) -> Dict[str, Any]:
        """Get information about chunking capabilities"""
        return {
            "semantic_chunking_enabled": self.semantic_chunking_enabled,
            "spacy_model": "en_core_sci_scibert" if self.semantic_chunking_enabled else None,
            "embedding_model": config.embedding_model,
            "supported_formats": self.get_supported_formats(),
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap
        }
    
    async def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Process all documents in a directory
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of all extracted chunks from all documents
        """
        all_chunks = []
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return []
        
        # Find all supported document files
        supported_extensions = ['.pdf', '.docx', '.pptx', '.xlsx', '.csv', '.md', '.html']
        document_files = []
        
        for ext in supported_extensions:
            document_files.extend(directory.glob(f"*{ext}"))
        
        logger.info(f"Found {len(document_files)} document files to process")
        
        for doc_file in document_files:
            try:
                logger.info(f"Processing: {doc_file.name}")
                result = self.process_document(str(doc_file))
                
                if result:
                    chunks = self.extract_chunks(result)
                    all_chunks.extend(chunks)
                    logger.info(f"✅ {doc_file.name}: {len(chunks)} chunks extracted")
                else:
                    logger.warning(f"❌ {doc_file.name}: Processing failed")
                    
            except Exception as e:
                logger.error(f"❌ {doc_file.name}: Error - {str(e)}")
                continue
        
        logger.info(f"Directory processing complete: {len(all_chunks)} total chunks from {len(document_files)} documents")
        return all_chunks