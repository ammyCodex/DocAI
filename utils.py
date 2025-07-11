import cohere
import numpy as np
import os
from pypdf import PdfReader
from docx import Document
import streamlit as st
import faiss

def get_document_text(files):
    text = ""
    for file in files:
        try:
            if file.name.lower().endswith('.pdf'):
                reader = PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() or ""
            elif file.name.lower().endswith('.docx'):
                doc = Document(file)
                for para in doc.paragraphs:
                    text += para.text + "\n"
            else:
                st.warning(f"Unsupported file type: {file.name}")
        except Exception as e:
            st.error(f"Failed to read {file.name}: {e}")
    return text

def get_chunked_text(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def get_faiss_index(chunks, cohere_api_key):
    co = cohere.Client(cohere_api_key)
    embeddings = []
    for chunk in chunks:
        emb = co.embed(texts=[chunk], model="embed-english-v3.0", input_type="search_document").embeddings[0]
        embeddings.append(emb)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def search_faiss_index(index, chunks, query, cohere_api_key, top_k=3):
    co = cohere.Client(cohere_api_key)
    query_emb = np.array(co.embed(texts=[query], model="embed-english-v3.0", input_type="search_query").embeddings).astype("float32")
    D, I = index.search(query_emb, top_k)
    return [chunks[i] for i in I[0]] 