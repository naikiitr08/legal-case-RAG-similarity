# # src/rag_processor.py

# import chromadb
# from chromadb.utils import embedding_functions
# from typing import List, Dict
# import os
# import json
# from sentence_transformers import SentenceTransformer
# import fitz  # PyMuPDF
# import logging

# logger = logging.getLogger(__name__)

# class LegalDocumentRAG:
#     def __init__(self, api_key: str, collection_name: str = "legal_cases"):
#         """Initialize RAG with simplified structure."""
#         self.api_key = api_key
        
#         # Initialize embedding model
#         self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
#         # Initialize ChromaDB
#         self.client = chromadb.PersistentClient(path="chroma_db")
#         self.collection = self.client.get_or_create_collection(
#             name=collection_name,
#             embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
#                 model_name='all-MiniLM-L6-v2'
#             )
#         )
        
#         # Separate storage for detailed metadata
#         self.metadata_dir = "document_metadata"
#         os.makedirs(self.metadata_dir, exist_ok=True)

#     def extract_text(self, pdf_path: str) -> str:
#         """Extract text from PDF."""
#         try:
#             with fitz.open(pdf_path) as doc:
#                 text = ""
#                 for page in doc:
#                     text += page.get_text()
#                 return text.strip()
#         except Exception as e:
#             logger.error(f"Failed to extract text from {pdf_path}: {str(e)}")
#             raise

#     def process_document(self, pdf_path: str) -> tuple[Dict, str]:
#         """Process document and return both semantic content and extracted text."""
#         # Extract text
#         text = self.extract_text(pdf_path)
        
#         # Basic metadata
#         simple_metadata = {
#             'filename': os.path.basename(pdf_path),
#             'path': pdf_path,
#             'text_length': len(text)
#         }
        
#         return simple_metadata, text

#     def add_to_rag(self, pdf_path: str) -> None:
#         """Add document to RAG system with simplified metadata."""
#         try:
#             filename = os.path.basename(pdf_path)
            
#             # Process document
#             metadata, text = self.process_document(pdf_path)
            
#             # Store in ChromaDB with simplified metadata
#             self.collection.add(
#                 documents=[text],
#                 metadatas=[metadata],
#                 ids=[filename]
#             )
            
#             logger.info(f"Successfully added {filename} to RAG database")
            
#         except Exception as e:
#             logger.error(f"Failed to add {pdf_path} to RAG: {str(e)}")
#             raise

#     def find_similar(self, query_pdf: str, top_k: int = 5) -> List[Dict]:
#         """Find similar documents using text similarity."""
#         try:
#             # Extract text from query PDF
#             query_text = self.extract_text(query_pdf)
            
#             # Query the collection
#             results = self.collection.query(
#                 query_texts=[query_text],
#                 n_results=top_k,
#                 include=["metadatas", "distances"]
#             )
            
#             similar_docs = []
#             if results['metadatas']:
#                 for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
#                     similarity_score = (1 - distance) * 100
#                     similar_docs.append({
#                         'filename': metadata['filename'],
#                         'similarity_score': round(similarity_score, 2),
#                         'path': metadata['path']
#                     })
            
#             return similar_docs
            
#         except Exception as e:
#             logger.error(f"Error finding similar documents: {str(e)}")
#             raise

# similarity with the petitioners issues 

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional
import os
import json
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import logging
import google.generativeai as genai
import time

logger = logging.getLogger(__name__)

# class LegalDocumentRAG:
#     def __init__(self, api_key: str, collection_name: str = "petitioner_issues"):
#         """Initialize RAG focusing on petitioner issues."""
#         self.api_key = api_key
        
#         # Initialize Gemini for extracting petitioner issues
#         genai.configure(api_key=api_key)
#         self.model = genai.GenerativeModel('gemini-pro')
        
#         # Initialize embedding model
#         self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
#         # Initialize ChromaDB specifically for petitioner issues
#         self.client = chromadb.PersistentClient(path="chroma_db")
#         self.collection = self.client.get_or_create_collection(
#             name=collection_name,
#             embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
#                 model_name='all-MiniLM-L6-v2'
#             )
#         )

#     def extract_petitioner_issues(self, text: str) -> str:
#         """Extract only petitioner issues from document text."""
#         prompt = """
#         Please extract ONLY the issues raised by the petitioner from this legal document.
#         Focus solely on the petitioner's arguments and claims.
        
#         Document text:
#         {text}
        
#         Return only the petitioner's issues in a clear, concise format.
#         """
        
#         try:
#             response = self.model.generate_content(prompt.format(text=text))
#             return response.text.strip()
#         except Exception as e:
#             logger.error(f"Failed to extract petitioner issues: {str(e)}")
#             raise

#     def extract_text(self, pdf_path: str) -> str:
#         """Extract text from PDF."""
#         try:
#             with fitz.open(pdf_path) as doc:
#                 text = ""
#                 for page in doc:
#                     text += page.get_text()
#                 return text.strip()
#         except Exception as e:
#             logger.error(f"Failed to extract text from {pdf_path}: {str(e)}")
#             raise

#     def process_document(self, pdf_path: str) -> tuple[Dict, str]:
#         """Process document focusing only on petitioner issues."""
#         # Extract text
#         text = self.extract_text(pdf_path)
        
#         # Extract petitioner issues
#         petitioner_issues = self.extract_petitioner_issues(text)
        
#         # Basic metadata
#         metadata = {
#             'filename': os.path.basename(pdf_path),
#             'path': pdf_path,
#             'full_text_length': len(text),
#             'petitioner_issues_length': len(petitioner_issues)
#         }
        
#         return metadata, petitioner_issues

#     def add_to_rag(self, pdf_path: str) -> None:
#         """Add document's petitioner issues to RAG system."""
#         try:
#             filename = os.path.basename(pdf_path)
            
#             # Process document to get only petitioner issues
#             metadata, petitioner_issues = self.process_document(pdf_path)
            
#             # Store in ChromaDB with petitioner issues only
#             self.collection.add(
#                 documents=[petitioner_issues],
#                 metadatas=[metadata],
#                 ids=[filename]
#             )
            
#             logger.info(f"Successfully added petitioner issues from {filename} to RAG database")
            
#         except Exception as e:
#             logger.error(f"Failed to add {pdf_path} to RAG: {str(e)}")
#             raise

#     def find_similar(self, query_pdf: str, top_k: int = 5) -> List[Dict]:
#         """Find documents with similar petitioner issues."""
#         try:
#             # Extract text and get petitioner issues from query PDF
#             query_text = self.extract_text(query_pdf)
#             query_petitioner_issues = self.extract_petitioner_issues(query_text)
            
#             # Query the collection using only petitioner issues
#             results = self.collection.query(
#                 query_texts=[query_petitioner_issues],
#                 n_results=top_k,
#                 include=["metadatas", "distances", "documents"]
#             )
            
#             similar_docs = []
#             if results['metadatas']:
#                 for metadata, distance, issues in zip(
#                     results['metadatas'][0], 
#                     results['distances'][0],
#                     results['documents'][0]
#                 ):
#                     similarity_score = (1 - distance) * 100
#                     similar_docs.append({
#                         'filename': metadata['filename'],
#                         'similarity_score': round(similarity_score, 2),
#                         'path': metadata['path'],
#                         'petitioner_issues': issues  # Include the matching issues
#                     })
            
#             return similar_docs
            
#         except Exception as e:
#             logger.error(f"Error finding similar documents: {str(e)}")
#             raise

class LegalDocumentRAG:
    def __init__(self, api_key: str, collection_name: str = "petitioner_issues"):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path="chroma_db")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name='all-MiniLM-L6-v2'
            )
        )

    def extract_text(self, pdf_path: str) -> str:
        """Extract text content from PDF file."""
        try:
            with fitz.open(pdf_path) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                return text.strip()
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {pdf_path}: {str(e)}")
            raise

    def extract_petitioner_issues(self, text: str) -> Optional[str]:
        """Extract petitioner issues with error handling and retries."""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(retry_delay * attempt)
                
                prompt = """
                Extract ONLY the main issues/arguments raised by the petitioner.
                Include only the key points, maximum 3-4 main issues.
                
                Text: {text}
                
                Return just the list of main petitioner issues, no additional text.
                """
                
                response = self.model.generate_content(prompt.format(text=text[:5000]))
                return response.text.strip()
            
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to extract petitioner issues after {max_retries} attempts")
                    return None
        return None

    def add_to_rag(self, pdf_path: str) -> None:
        """Add document to RAG in batches."""
        try:
            filename = os.path.basename(pdf_path)
            text = self.extract_text(pdf_path)
            
            # Extract petitioner issues
            petitioner_issues = self.extract_petitioner_issues(text)
            if not petitioner_issues:
                logger.warning(f"Skipping {filename} - could not extract petitioner issues")
                return
            
            metadata = {
                'filename': filename,
                'path': pdf_path,
                'processed_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Add to collection
            self.collection.add(
                documents=[petitioner_issues],
                metadatas=[metadata],
                ids=[filename]
            )
            
            logger.info(f"Successfully processed {filename}")
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {str(e)}")

    def find_similar(self, query_pdf: str, top_k: int = 5) -> List[Dict]:
        """Find similar documents based on petitioner issues."""
        try:
            # Extract query document's petitioner issues
            query_text = self.extract_text(query_pdf)
            query_issues = self.extract_petitioner_issues(query_text)
            
            if not query_issues:
                logger.error("Could not extract petitioner issues from query document")
                return []
            
            # Get similar documents
            results = self.collection.query(
                query_texts=[query_issues],
                n_results=top_k,
                include=["metadatas", "distances", "documents"]
            )
            
            similar_docs = []
            if results['distances'] and results['distances'][0]:
                for metadata, distance, issues in zip(
                    results['metadatas'][0],
                    results['distances'][0],
                    results['documents'][0]
                ):
                    similarity_score = (1 - distance) * 100
                  #  if similarity_score > 20:  # Only include if similarity is above 20%
                    similar_docs.append({
                            'filename': metadata['filename'],
                            'similarity_score': round(similarity_score, 2),
                            'petitioner_issues': issues
                        })
                
                # Sort by similarity score
                similar_docs.sort(key=lambda x: x['similarity_score'], reverse=True)
                similar_docs = similar_docs[:5]  # Keep only top 5
            
            return similar_docs
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []

