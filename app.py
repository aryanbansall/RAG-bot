import streamlit as st
import os
import time
import tempfile
import hashlib
from dotenv import load_dotenv
from pymongo import MongoClient

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Document QA Assistant (MongoDB)",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    .file-badge {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 6px 12px;
        margin: 4px 0;
        font-size: 0.85em;
        color: #94A3B8;
    }
    .source-box {
        background-color: #1E293B;
        border-left: 4px solid #10B981;
        padding: 10px 15px;
        margin: 8px 0;
        border-radius: 4px;
        font-size: 0.9em;
        color: #E2E8F0;
    }
    .status-badge-active {
        color: #10B981;
        font-weight: 600;
    }
    .status-badge-inactive {
        color: #EF4444;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to get secrets safely from Streamlit Secrets or Environment
def get_secret(key_name, fallback_key=None):
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
        if fallback_key and fallback_key in st.secrets:
            return st.secrets[fallback_key]
    except Exception:
        pass
    val = os.getenv(key_name)
    if val:
        return val
    if fallback_key:
        return os.getenv(fallback_key) or ""
    return ""

# Retrieve configured keys
default_groq_key = get_secret("API_KEY", "GROQ_API_KEY")
default_google_key = get_secret("GOOGLE_API_KEY")
default_mongo_uri = get_secret("MONGODB_URI") or "mongodb+srv://aryanbansalcontact_db_user:57W6n7LTFPh5A3fu@backend-learn.srz3k5n.mongodb.net/"
mongo_db_name = get_secret("MONGODB_DB_NAME") or "RAG-bot"
mongo_collection_name = "vector_index"

# Initialize Session State Variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectors" not in st.session_state:
    st.session_state.vectors = None

if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

# MongoDB Connection Helper
@st.cache_resource(show_spinner=False)
def get_mongo_collection(uri, db_name, coll_name):
    if not uri:
        return None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=4000)
        db = client[db_name]
        return db[coll_name]
    except Exception:
        return None

mongo_coll = get_mongo_collection(default_mongo_uri, mongo_db_name, mongo_collection_name)

# Sidebar Section
with st.sidebar:
    st.title("🍃 RAG Workspace (MongoDB)")
    st.caption(f"Cluster: `backend-learn.srz3k5n` | Database: `{mongo_db_name}`")
    
    st.markdown("---")
    
    # API Credentials Section
    st.subheader("🔑 API Credentials")
    if default_groq_key and default_google_key:
        st.success("🔒 API Keys loaded from Server Environment.")
        with st.expander("🔑 Override API Keys (Optional)"):
            groq_api_key_input = st.text_input("Groq API Key", value=default_groq_key, type="password")
            google_api_key_input = st.text_input("Google Gemini API Key", value=default_google_key, type="password")
    else:
        groq_api_key_input = st.text_input("Groq API Key", value=default_groq_key, type="password")
        google_api_key_input = st.text_input("Google Gemini API Key", value=default_google_key, type="password")
    
    st.markdown("---")
    
    # Model & Retrieval Settings
    st.subheader("⚙️ Model Settings")
    model_choice = st.selectbox(
        "Groq LLM Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
        index=0
    )
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    k_retrieved = st.slider("Retrieved Chunks (k)", 1, 10, 6, 1)
    
    st.markdown("---")
    
    # Document Ingestion Section
    st.subheader("📥 Document Ingestion to MongoDB")
    load_assets = st.checkbox("Include `./Assets` directory PDFs", value=True)
    
    assets_files = []
    if os.path.exists("./Assets"):
        assets_files = [f for f in os.listdir("./Assets") if f.endswith(".pdf")]
    
    if assets_files:
        st.caption(f"📁 Found {len(assets_files)} PDF(s) in `./Assets`:")
        for f in assets_files:
            st.markdown(f"<div class='file-badge'>📄 {f}</div>", unsafe_allow_html=True)
            
    uploaded_files = st.file_uploader("Upload Custom PDF Documents", type=["pdf"], accept_multiple_files=True)
    
    with st.expander("🛠️ Advanced Chunking Settings"):
        chunk_size = st.number_input("Chunk Size", min_value=100, max_value=4000, value=1000, step=100)
        chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=1000, value=200, step=50)

    # Smart Incremental Ingestion to MongoDB with Rate Limit Retry & Deduplication
    def create_and_store_in_mongodb(show_toast=True):
        if not google_api_key_input:
            st.error("Google API Key is required to generate vector embeddings!")
            return False

        docs = []
        loaded_file_names = []
        
        if load_assets and os.path.exists("./Assets"):
            try:
                assets_loader = PyPDFDirectoryLoader("./Assets")
                assets_docs = assets_loader.load()
                docs.extend(assets_docs)
                loaded_file_names.extend(assets_files)
            except Exception as e:
                st.warning(f"Could not load documents from ./Assets: {e}")
        
        if uploaded_files:
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    loader = PyPDFLoader(tmp_path)
                    uploaded_docs = loader.load()
                    for d in uploaded_docs:
                        d.metadata["source"] = file.name
                    docs.extend(uploaded_docs)
                    loaded_file_names.append(file.name)
                except Exception as e:
                    st.error(f"Error reading {file.name}: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

        if not docs:
            st.error("No valid PDF documents found to ingest!")
            return False

        with st.spinner("Processing documents & checking MongoDB cache..."):
            try:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                final_documents = text_splitter.split_documents(docs)
                embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=google_api_key_input)
                
                # Fetch existing cached vector hashes from MongoDB
                existing_records = {}
                if mongo_coll is not None:
                    try:
                        for r in mongo_coll.find({}, {"doc_id": 1, "text": 1, "metadata": 1, "embedding": 1}):
                            if "doc_id" in r:
                                existing_records[r["doc_id"]] = r
                    except Exception:
                        pass

                chunks_to_embed = []
                cached_tuples = []

                for doc in final_documents:
                    doc_id = hashlib.sha256(f"{doc.page_content}:{doc.metadata.get('source')}:{doc.metadata.get('page')}".encode()).hexdigest()
                    if doc_id in existing_records:
                        rec = existing_records[doc_id]
                        cached_tuples.append((rec["text"], rec["embedding"], rec["metadata"]))
                    else:
                        chunks_to_embed.append((doc_id, doc))

                # Batch embed new chunks with 429 Rate Limit backoff retry
                batch_size = 10
                new_tuples = []
                
                if chunks_to_embed:
                    progress_text = st.empty()
                    for i in range(0, len(chunks_to_embed), batch_size):
                        batch = chunks_to_embed[i:i+batch_size]
                        batch_texts = [item[1].page_content for item in batch]
                        progress_text.info(f"⚡ Generating embeddings for new chunks ({i+1}-{min(i+batch_size, len(chunks_to_embed))}/{len(chunks_to_embed)})...")
                        
                        for attempt in range(5):
                            try:
                                vecs = embeddings.embed_documents(batch_texts)
                                mongo_inserts = []
                                for (doc_id, doc), vec in zip(batch, vecs):
                                    record = {
                                        "doc_id": doc_id,
                                        "text": doc.page_content,
                                        "metadata": doc.metadata,
                                        "embedding": vec
                                    }
                                    mongo_inserts.append(record)
                                    new_tuples.append((doc.page_content, vec, doc.metadata))
                                
                                if mongo_coll is not None and mongo_inserts:
                                    mongo_coll.insert_many(mongo_inserts)
                                time.sleep(0.5)  # Stay within Google API Rate Limit
                                break
                            except Exception as e:
                                err_str = str(e)
                                if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < 4:
                                    progress_text.warning(f"⏳ Rate limit hit. Pausing for 5 seconds before retrying batch {i//batch_size + 1}... (Attempt {attempt+1}/5)")
                                    time.sleep(5)
                                else:
                                    raise e

                    progress_text.empty()

                all_tuples = cached_tuples + new_tuples
                text_embeddings = [(t[0], t[1]) for t in all_tuples]
                metadatas = [t[2] for t in all_tuples]
                
                vector_store = FAISS.from_embeddings(text_embeddings=text_embeddings, embedding=embeddings, metadatas=metadatas)
                
                st.session_state.vectors = vector_store
                st.session_state.doc_count = len(docs)
                st.session_state.chunk_count = len(all_tuples)
                st.session_state.indexed_files = list(set(loaded_file_names))
                
                if show_toast:
                    st.success(f"🍃 Ready! {len(cached_tuples)} chunks reused from cache, {len(new_tuples)} new chunks embedded into MongoDB.")
                return True
            except Exception as e:
                st.error(f"Error during ingestion: {e}")
                return False

    # Function to load existing vector embeddings directly from MongoDB
    def load_from_mongodb():
        if mongo_coll is None or not google_api_key_input:
            return False
        
        try:
            records = list(mongo_coll.find({}))
            if not records:
                return False
            
            embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=google_api_key_input)
            text_embeddings = [(r["text"], r["embedding"]) for r in records]
            metadatas = [r["metadata"] for r in records]
            vector_store = FAISS.from_embeddings(text_embeddings=text_embeddings, embedding=embeddings, metadatas=metadatas)
            
            files = list(set(m.get("source", "Document") for m in metadatas))
            st.session_state.vectors = vector_store
            st.session_state.doc_count = len(records)
            st.session_state.chunk_count = len(records)
            st.session_state.indexed_files = [os.path.basename(f) for f in files]
            return True
        except Exception as e:
            st.warning(f"⚠️ Could not read from MongoDB cluster: {e}. Falling back to local ingestion.")
            return False

    if st.button("⚡ Ingest & Store in MongoDB", use_container_width=True):
        create_and_store_in_mongodb(show_toast=True)
        
    st.markdown("---")
    
    # System Status & Reset
    st.subheader("📊 MongoDB Status")
    if st.session_state.vectors is not None:
        st.markdown(f"**Status:** <span class='status-badge-active'>🟢 Index Active</span>", unsafe_allow_html=True)
        st.write(f"🍃 **Database:** `{mongo_db_name}`")
        st.write(f"📄 **Chunks Loaded:** {st.session_state.chunk_count}")
        if st.session_state.indexed_files:
            st.caption("Indexed files:")
            for fname in st.session_state.indexed_files:
                st.caption(f"• `{fname}`")
    else:
        st.markdown(f"**Status:** <span class='status-badge-inactive'>🔴 Not Loaded</span>", unsafe_allow_html=True)
        
    if st.button("🗑️ Clear MongoDB Database & Chat", use_container_width=True):
        if mongo_coll is not None:
            try:
                mongo_coll.delete_many({})
            except Exception:
                pass
        st.session_state.vectors = None
        st.session_state.messages = []
        st.session_state.doc_count = 0
        st.session_state.chunk_count = 0
        st.session_state.indexed_files = []
        st.rerun()

# Auto-load existing vectors from MongoDB on app launch, or ingest if database is empty
if st.session_state.vectors is None and google_api_key_input:
    loaded = load_from_mongodb()
    if not loaded:
        with st.spinner("Ingesting documents into workspace..."):
            create_and_store_in_mongodb(show_toast=False)

# Main Interface Header
st.title("💬 Document Question Answering System")
st.caption("Powered by LangChain + MongoDB Atlas Vector Store + Groq Llama 3.1 + Gemini Embeddings")

# Display status banner
if st.session_state.vectors is None:
    st.warning("⚠️ **Vector index is empty.** Click **'⚡ Ingest & Store in MongoDB'** to load documents.")
else:
    st.info(f"🍃 **Ready:** Loaded {st.session_state.chunk_count} vector embeddings from database **'{mongo_db_name}'**!")

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if "response_time" in message:
            st.caption(f"⏱️ Response time: {message['response_time']:.2f} seconds")
            
        if "sources" in message and message["sources"]:
            with st.expander(f"🔍 View {len(message['sources'])} Retrieved Context Passages"):
                for i, doc in enumerate(message["sources"]):
                    source_name = doc.metadata.get("source", "Document")
                    page_num = doc.metadata.get("page", 0) + 1
                    st.markdown(f"**Source {i+1}:** `{os.path.basename(source_name)}` (Page {page_num})")
                    st.markdown(f"<div class='source-box'>{doc.page_content}</div>", unsafe_allow_html=True)

# Chat Input & RAG Pipeline Execution
user_query = st.chat_input("Ask a question about your documents...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if not groq_api_key_input:
            st.error("Missing Groq API Key! Please enter it in the sidebar.")
        elif st.session_state.vectors is None:
            success = load_from_mongodb() or create_and_store_in_mongodb(show_toast=False)
            if not success:
                st.error("Please click '⚡ Ingest & Store in MongoDB' in the sidebar to index your documents.")
                st.stop()
        
        if st.session_state.vectors is not None:
            try:
                with st.spinner("Retrieving passages & generating answer..."):
                    llm = ChatGroq(
                        groq_api_key=groq_api_key_input,
                        model=model_choice,
                        temperature=temperature
                    )

                    prompt_template = """
                    You are an intelligent document question-answering assistant.
                    Answer the user's question accurately and thoroughly based on the provided document context passages.
                    Be sure to synthesize code samples, mathematical formulas, algorithms, and textual explanations present in the context.
                    If the provided context does not contain any relevant information to answer the question, clearly state that the provided context does not contain sufficient details.

                    <context>
                    {context}
                    </context>

                    Question: {input}
                    Answer:
                    """
                    prompt = ChatPromptTemplate.from_template(prompt_template)

                    retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": k_retrieved})
                    retrieved_docs = retriever.invoke(user_query)

                    context_text = "\n\n---\n\n".join(
                        f"[Document: {os.path.basename(doc.metadata.get('source', 'Unknown'))} | Page {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}"
                        for doc in retrieved_docs
                    )
                    
                    formatted_prompt = prompt.invoke({"context": context_text, "input": user_query})

                    start_time = time.time()
                    response = llm.invoke(formatted_prompt)
                    elapsed_time = time.time() - start_time

                    answer = response.content

                    st.markdown(answer)
                    st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")

                    if retrieved_docs:
                        with st.expander(f"🔍 View {len(retrieved_docs)} Retrieved Context Passages"):
                            for i, doc in enumerate(retrieved_docs):
                                source_name = doc.metadata.get("source", "Document")
                                page_num = doc.metadata.get("page", 0) + 1
                                st.markdown(f"**Source {i+1}:** `{os.path.basename(source_name)}` (Page {page_num})")
                                st.markdown(f"<div class='source-box'>{doc.page_content}</div>", unsafe_allow_html=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "response_time": elapsed_time,
                        "sources": retrieved_docs
                    })

            except Exception as e:
                st.error(f"Error during query execution: {e}")