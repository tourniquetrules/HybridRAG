import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.document import ConversionResult
from docling.datamodel.base_models import InputFormat, ConversionStatus

from config import config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document processing using Docling for emergency medicine PDFs"""
    
    def __init__(self):
        """Initialize the document processor with Docling"""
        # Configure pipeline for high-quality medical document processing
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True  # Enable OCR for scanned documents
        pipeline_options.do_table_structure = True  # Extract table structure
        pipeline_options.table_structure_options.do_cell_matching = True
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
    def process_document(self, file_path: str) -> Optional[ConversionResult]:
        """
        Process a single document using Docling
        
        Args:
            file_path: Path to the document file
            
        Returns:
            ConversionResult containing processed document or None if failed
        """
        try:
            logger.info(f"Processing document: {file_path}")
            
            # Convert document using Docling
            result = self.converter.convert(file_path)
            
            # Check conversion status using Docling v2 API
            if result.status in [ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS]:
                logger.info(f"Successfully processed: {file_path}")
                return result
            else:
                logger.error(f"Failed to process {file_path}: {result.status}")
                if hasattr(result.status, 'errors'):
                    logger.error(f"Errors: {result.status.errors}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return None
    
    def extract_text_chunks(self, conversion_result: ConversionResult) -> List[Dict[str, Any]]:
        """
        Extract text chunks from a processed document
        
        Args:
            conversion_result: Result from Docling conversion
            
        Returns:
            List of text chunks with metadata
        """
        chunks = []
        
        try:
            document = conversion_result.document
            
            # Extract main text content
            main_text = document.export_to_markdown()
            
            # Split into chunks
            text_chunks = self._split_text(main_text, config.chunk_size, config.chunk_overlap)
            
            # Extract metadata
            metadata = {
                "source": str(conversion_result.input.file),
                "title": getattr(document, 'name', "Unknown"),
                "num_pages": len(document.pages) if hasattr(document, 'pages') else 0,
                "conversion_time": getattr(conversion_result.status, 'processing_time', 0)
            }
            
            # Create chunks with metadata
            for i, chunk_text in enumerate(text_chunks):
                chunk = {
                    "text": chunk_text,
                    "chunk_id": f"{Path(conversion_result.input.file).stem}_chunk_{i}",
                    "metadata": {
                        **metadata,
                        "chunk_index": i,
                        "total_chunks": len(text_chunks)
                    }
                }
                chunks.append(chunk)
                
            # Extract tables as separate chunks
            table_chunks = self._extract_table_chunks(document, metadata)
            chunks.extend(table_chunks)
            
            logger.info(f"Extracted {len(chunks)} chunks from document")
            return chunks
            
        except Exception as e:
            logger.error(f"Error extracting chunks: {str(e)}")
            return []
    
    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks"""
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
        """Extract tables as separate chunks"""
        table_chunks = []
        
        try:
            # Access tables from the DoclingDocument
            for i, table in enumerate(document.tables):
                    # Convert table to markdown format
                    table_text = self._table_to_markdown(table)
                    
                    if table_text.strip():
                        chunk = {
                            "text": table_text,
                            "chunk_id": f"{Path(base_metadata['source']).stem}_table_{i}",
                            "metadata": {
                                **base_metadata,
                                "chunk_type": "table",
                                "table_index": i
                            }
                        }
                        table_chunks.append(chunk)
                        
        except Exception as e:
            logger.warning(f"Error extracting tables: {str(e)}")
            
        return table_chunks
    
    def _table_to_markdown(self, table: Any) -> str:
        """Convert table object to markdown format"""
        try:
            # This is a simplified conversion - adjust based on Docling's table structure
            if hasattr(table, 'to_markdown'):
                return table.to_markdown()
            elif hasattr(table, 'data'):
                # Convert table data to markdown manually
                rows = table.data
                if not rows:
                    return ""
                
                # Create header
                header = "| " + " | ".join(str(cell) for cell in rows[0]) + " |"
                separator = "| " + " | ".join("---" for _ in rows[0]) + " |"
                
                # Create data rows
                data_rows = []
                for row in rows[1:]:
                    row_text = "| " + " | ".join(str(cell) for cell in row) + " |"
                    data_rows.append(row_text)
                
                return "\n".join([header, separator] + data_rows)
            else:
                return str(table)
                
        except Exception as e:
            logger.warning(f"Error converting table to markdown: {str(e)}")
            return str(table)
    
    async def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Process all PDF files in a directory
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            List of all extracted chunks
        """
        all_chunks = []
        pdf_files = list(Path(directory_path).glob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            try:
                result = self.process_document(str(pdf_file))
                if result:
                    chunks = self.extract_text_chunks(result)
                    all_chunks.extend(chunks)
                    
                    # Limit total documents processed
                    if len(all_chunks) >= config.max_documents:
                        logger.info(f"Reached maximum document limit ({config.max_documents})")
                        break
                        
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
                continue
        
        logger.info(f"Processed {len(pdf_files)} files, extracted {len(all_chunks)} chunks")
        return all_chunks