import cohere
import numpy as np
import os
from pypdf import PdfReader
from docx import Document
import streamlit as st
import faiss


def get_document_text(files):
    """Extract text from PDF and DOCX files."""
    text = ""
    for file in files:
        try:
            if file.name.lower().endswith('.pdf'):
                reader = PdfReader(file)
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text += f"[Page {page_num + 1}]\n{page_text}\n"
            elif file.name.lower().endswith('.docx'):
                doc = Document(file)
                for para in doc.paragraphs:
                    if para.text.strip():
                        text += para.text + "\n"
            else:
                st.warning(f"Unsupported file type: {file.name}")
        except Exception as e:
            st.error(f"Failed to read {file.name}: {e}")
    return text


def get_chunked_text(text, chunk_size=600, chunk_overlap=300):
    """
    Split text into overlapping chunks for better RAG retrieval.
    Uses 50% overlap (300/600) for better semantic context preservation.
    """
    chunks = []
    if not text or not text.strip():
        return chunks
    
    # Split by sentences first for better semantic chunks
    text = text.strip()
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # Try to break at a logical boundary (sentence/paragraph)
        if end < len(text):
            # Look for last period or newline within the chunk
            last_period = text.rfind(".", start, end)
            last_newline = text.rfind("\n", start, end)
            last_break = max(last_period, last_newline)
            
            if last_break > start + (chunk_size // 2):  # Only if it's not too early
                end = last_break + 1
        
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        start += chunk_size - chunk_overlap
    
    return chunks


def get_faiss_index(chunks, cohere_api_key):
    """
    Build FAISS index from document chunks using Cohere embeddings.
    Uses the latest embedding model for optimal performance.
    """
    if not chunks:
        raise ValueError("No chunks provided for indexing")
    
    co = cohere.Client(cohere_api_key)
    embeddings = []
    
    try:
        # Use v3 embedding model optimized for semantic search
        # Process in batches for better performance
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_resp = co.embed(
                texts=batch,
                model="embed-english-v3.0",
                input_type="search_document"
            )
            embeddings.extend(batch_resp.embeddings)
        
        if not embeddings:
            raise ValueError("Failed to generate embeddings")
        
        # Convert to numpy array for FAISS
        embeddings = np.array(embeddings).astype("float32")
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        return index, embeddings
        
    except Exception as e:
        raise Exception(f"Error creating FAISS index: {str(e)}")


def search_faiss_index(index, chunks, query, cohere_api_key, top_k=3):
    """
    Search FAISS index for relevant chunks using semantic similarity.
    Returns the top-k most relevant chunks for RAG context.
    """
    if not chunks or not query:
        return []
    
    co = cohere.Client(cohere_api_key)
    
    try:
        # Embed the query with search_query input type
        query_resp = co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        
        if not query_resp.embeddings:
            raise ValueError("Failed to embed query")
        
        query_emb = np.array(query_resp.embeddings).astype("float32")
        
        # Search FAISS index
        distances, indices = index.search(query_emb, min(top_k, len(chunks)))
        
        # Return chunks as a list, filtering out invalid indices
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(chunks):
                results.append(chunks[idx])
        
        return results
        
    except Exception as e:
        raise Exception(f"Error searching FAISS index: {str(e)}") 