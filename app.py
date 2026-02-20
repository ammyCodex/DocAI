import streamlit as st
import time
from datetime import datetime
import streamlit.components.v1 as components
import cohere
from utils import get_document_text, get_chunked_text, get_faiss_index, search_faiss_index


def get_cohere_response(question, context, cohere_api_key, model="command-a-03-2025"):
    """
    Generate a response using Cohere's Chat API with RAG context.
    Uses command-a-03-2025 for optimal RAG performance.
    """
    co = cohere.Client(cohere_api_key)
    
    # Validate inputs
    question = (question or "").strip()
    context = (context or "").strip()
    
    if not question:
        raise ValueError("Empty question provided")
    
    # RAG system prompt - optimized for document Q&A
    system_prompt = (
        "You are an expert document analyst. Based on the provided context, "
        "answer the user's question accurately and concisely. "
        "If the answer is not found in the context, say 'This information is not available in the document.'"
    )
    
    # Build the RAG prompt with clear separation
    rag_context = f"Document Context:\n{context}\n" if context else "No context available.\n"
    
    try:
        # Build the complete user message with context and question
        user_message = f"{system_prompt}\n\n{rag_context}\nQuestion: {question}"
        
        # Use Cohere's chat method with RAG-optimized parameters
        response = co.chat(
            model=model,
            message=user_message,
            max_tokens=150,  # Concise answers
            temperature=0.3,  # Lower temp for consistency in factual QA
        )
        
        # Extract text from response
        if hasattr(response, "text"):
            return response.text.strip()
        elif isinstance(response, dict) and "text" in response:
            return response["text"].strip()
        elif hasattr(response, "message"):
            if isinstance(response.message, dict) and "content" in response.message:
                return response.message["content"].strip()
            return str(response.message).strip()
        else:
            # Fallback for different response structures
            return str(response).strip()
            
    except Exception as e:
        # Log error and raise with context
        raise Exception(f"Cohere API error: {str(e)}")


def main():
    st.set_page_config(
        page_title='PDF & DOCX ChatBot',
        page_icon=':sparkles:',
        layout='wide',
        initial_sidebar_state='auto'
    )
    
    st.markdown('''
        <style>
        body, .stApp { background: #181c24 !important; }
        .sticky-sidebar {
            position: sticky;
            top: 1.5rem;
            max-height: 85vh;
            overflow-y: auto;
            overflow-x: hidden;
            background: #23263a;
            border-radius: 14px;
            padding: 1rem 0.7rem 1rem 0.7rem;
            box-shadow: 0 2px 12px 0 rgba(67,70,84,0.10);
        }
        .sticky-sidebar::-webkit-scrollbar {
            width: 6px;
        }
        .sticky-sidebar::-webkit-scrollbar-track {
            background: #1a1e28;
            border-radius: 10px;
        }
        .sticky-sidebar::-webkit-scrollbar-thumb {
            background: #4f8bf9;
            border-radius: 10px;
            border: 2px solid #23263a;
        }
        .sticky-sidebar::-webkit-scrollbar-thumb:hover {
            background: #6fa0ff;
        }
        .qa-group {
            background: #23263a;
            border: 1.5px solid #434654;
            border-radius: 14px;
            padding: 0.7rem 0.9rem 0.7rem 0.9rem;
            margin-bottom: 1.1rem;
            transition: box-shadow 0.3s, background 0.3s;
        }
        .qa-group:hover {
            background: linear-gradient(135deg, #23263a 60%, #434654 100%);
            box-shadow: 0 6px 24px 0 rgba(79,139,249,0.18);
            border-color: #4f8bf9;
        }
        .chat-msg {
            margin-bottom: 1.1rem;
            display: flex;
            align-items: flex-start;
            gap: 0.7rem;
        }
        .chat-avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: #ffe066;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3rem;
            font-weight: bold;
            color: #23263a;
            flex-shrink: 0;
            border: 2.5px solid #fff;
            box-shadow: 0 2px 8px 0 rgba(255,224,102,0.10);
            transition: border 0.3s, box-shadow 0.3s;
        }
        .chat-avatar-bot {
            background: #4f8bf9;
            color: #fff;
            border: 2.5px solid #ffe066;
            box-shadow: 0 2px 8px 0 rgba(79,139,249,0.10);
        }
        .chat-bubble {
            background: #23263a;
            color: #ffe066;
            border-radius: 10px;
            padding: 0.7rem 1rem;
            font-size: 1rem;
            font-weight: 500;
            box-shadow: 0 1px 4px 0 rgba(67,70,84,0.10);
            margin-bottom: 0.2rem;
            max-width: 220px;
            word-break: break-word;
            transition: background 0.3s, color 0.3s;
        }
        .chat-bubble-bot {
            background: #434654;
            color: #fff;
        }
        .chat-timestamp {
            font-size: 0.82rem;
            color: #b0b0b0;
            margin-top: 0.1rem;
            margin-left: 2.5rem;
        }
        </style>
    ''', unsafe_allow_html=True)

    # Initialize session state
    if 'faiss_index' not in st.session_state:
        st.session_state['faiss_index'] = None
    if 'chunks' not in st.session_state:
        st.session_state['chunks'] = None
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'document_loaded' not in st.session_state:
        st.session_state['document_loaded'] = False

    if st.button("🧹 Clear Chat History"):
        st.session_state['faiss_index'] = None
        st.session_state['chunks'] = None
        st.session_state['chat_history'] = []
        st.session_state['document_loaded'] = False
        message = st.success("Chat history cleared!", icon="✅")
        time.sleep(2)
        message.empty()

    # Sidebar: header, description, and document upload/processing
    with st.sidebar:
        st.markdown('''
            <div style="background: linear-gradient(90deg, #23263a 0%, #434654 100%); padding: 1.5rem 1rem 1.2rem 1rem; border-radius: 0 0 20px 20px; margin-bottom: 1.5rem; box-shadow: 0 4px 16px 0 rgba(67,70,84,0.15);">
                <h2 style="color: #fff; font-size: 1.8rem; font-weight: 800; margin-bottom: 0.2rem; letter-spacing: -0.5px;">
                    <span style="vertical-align: middle;">🧠✨</span> DocAI
                </h2>
                <div style="color: #ffe066; font-size: 1rem; font-weight: 500; max-width: 300px;">
                    Smart document Q&A powered by RAG
                </div>
            </div>
        ''', unsafe_allow_html=True)
        
        st.subheader("📄 Upload & Process Document")
        docs = st.file_uploader(
            "Upload PDF or DOCX file(s)",
            accept_multiple_files=True,
            type=["pdf", "docx"]
        )
        
        # Get API key
        try:
            cohere_api_key = st.secrets["COHERE_API_KEY"]
        except KeyError:
            st.error("❌ COHERE_API_KEY not found in secrets. Please add it to .streamlit/secrets.toml")
            st.stop()
        
        if st.button("⚙️ Process Documents") and docs:
            with st.spinner("Processing documents..."):
                try:
                    raw_text = get_document_text(docs)
                    if not raw_text.strip():
                        st.error("No text could be extracted from the documents")
                    else:
                        chunks = get_chunked_text(raw_text)
                        st.info(f"Created {len(chunks)} chunks from {len(raw_text)} characters")
                        
                        index, embeddings = get_faiss_index(chunks, cohere_api_key)
                        st.session_state['faiss_index'] = index
                        st.session_state['chunks'] = chunks
                        st.session_state['document_loaded'] = True
                        st.success(f"✅ Documents processed! {len(chunks)} chunks indexed.")
                except Exception as e:
                    st.error(f"❌ Error processing documents: {e}")

    # Main area: center for chat, right for chat history
    center, right = st.columns([2, 1])
    
    with center:
        st.subheader("💬 Chat with Your Document")
        
        if not st.session_state['document_loaded']:
            st.info("👈 Upload and process a document to get started")
        else:
            question = st.text_input('Ask a question about your document:')
            
            if question:
                with st.spinner("Thinking..."):
                    try:
                        # Retrieve top 3 relevant chunks
                        top_chunks = search_faiss_index(
                            st.session_state['faiss_index'],
                            st.session_state['chunks'],
                            question,
                            cohere_api_key,
                            top_k=3
                        )
                        
                        if not top_chunks:
                            st.warning("No relevant content found in document")
                        else:
                            context = "\n---\n".join(top_chunks)
                            user_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            
                            # Generate response
                            response = get_cohere_response(question, context, cohere_api_key)
                            bot_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            
                            # Add to history
                            st.session_state['chat_history'].append({
                                'question': question,
                                'answer': response,
                                'user_time': user_time,
                                'bot_time': bot_time
                            })
                            
                            # Display with typewriter effect
                            st.markdown("### Answer:")
                            placeholder = st.empty()
                            typed = ""
                            for char in response:
                                typed += char
                                placeholder.markdown(typed)
                                time.sleep(0.01)
                    
                    except Exception as e:
                        st.error(f"❌ Error generating response: {e}")
    
    with right:
        st.markdown('<div class="sticky-sidebar">', unsafe_allow_html=True)
        st.subheader("📚 Chat History")
        
        if st.session_state['chat_history']:
            for entry in reversed(st.session_state['chat_history']):
                user_time = entry.get('user_time', '')
                bot_time = entry.get('bot_time', '')
                st.markdown(f'''
                <div class="qa-group">
                    <div class="chat-msg">
                        <div class="chat-avatar">🧑</div>
                        <div>
                            <div class="chat-bubble">{entry["question"]}</div>
                            <div class="chat-timestamp">{user_time}</div>
                        </div>
                    </div>
                    <div class="chat-msg">
                        <div class="chat-avatar chat-avatar-bot">🤖</div>
                        <div>
                            <div class="chat-bubble chat-bubble-bot">{entry["answer"]}</div>
                            <div class="chat-timestamp">{bot_time}</div>
                        </div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.info('No chat history yet.')
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()