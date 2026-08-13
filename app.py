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

# Page configuration - Clean Text Only
st.set_page_config(
    page_title="Your Guide in IIITD - Ashish",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Manage One-Time IIITD Splash Loader on Session Startup
if "has_seen_splash" not in st.session_state:
    st.session_state["has_seen_splash"] = True
    show_splash = True
else:
    show_splash = False

# Render Splash Screen Overlay ONLY ONCE on session startup
splash_html = """
    <div class="splash-overlay">
      <div class="iiitd-container">
        <span class="iiitd-char char-i1">I</span>
        <span class="iiitd-char char-i2">I</span>
        <span class="iiitd-char char-i3">I</span>
        <span class="iiitd-char char-t">T</span>
        <span class="iiitd-char char-d">D</span>
      </div>
    </div>
""" if show_splash else ""

splash_css = """
    /* IIITD Fullscreen Welcome Splash Screen Overlay */
    .splash-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-color: #050811;
        z-index: 9999999;
        display: flex;
        justify-content: center;
        align-items: center;
        animation: fadeOutSplash 0.45s cubic-bezier(0.4, 0, 0.2, 1) 1.8s forwards;
        pointer-events: none;
    }

    .iiitd-container {
        display: flex;
        align-items: center;
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', system-ui, sans-serif;
        font-size: 4.2rem;
        font-weight: 800;
        letter-spacing: 0.15em;
        color: #FFFFFF;
    }

    .iiitd-char {
        display: inline-block;
        opacity: 0;
        transform: translateY(12px);
    }

    .char-i1 {
        animation: revealChar 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) 0.1s forwards, moveI1 0.3s ease 0.4s forwards;
    }

    .char-i2 {
        animation: revealChar 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) 0.4s forwards, moveI2 0.3s ease 0.7s forwards;
    }

    .char-i3 {
        animation: revealChar 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) 0.7s forwards;
    }

    .char-t {
        animation: revealChar 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) 1.0s forwards;
        color: #6366F1;
    }

    .char-d {
        animation: revealChar 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) 1.25s forwards;
        color: #6366F1;
    }

    @keyframes revealChar {
        0% { opacity: 0; transform: scale(0.7) translateY(12px); }
        100% { opacity: 1; transform: scale(1) translateY(0); }
    }

    @keyframes moveI1 {
        0% { transform: translateX(0); }
        100% { transform: translateX(-14px); }
    }

    @keyframes moveI2 {
        0% { transform: translateX(0); }
        100% { transform: translateX(-7px); }
    }

    @keyframes fadeOutSplash {
        0% { opacity: 1; visibility: visible; }
        99% { opacity: 0; visibility: visible; }
        100% { opacity: 0; visibility: hidden; display: none; }
    }
""" if show_splash else ""

st.markdown(f"""
    <style>
    {splash_css}

    /* Global App Theme */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 3.5rem;
        max-width: 900px;
    }}
    .stApp {{
        background-color: #090D16;
        color: #F1F5F9;
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif;
    }}
    
    /* Ensure Sidebar Collapse / Uncollapse Toggle Control is ALWAYS Visible & Clickable */
    [data-testid="collapsedControl"] {{
        display: flex !important;
        visibility: visible !important;
        top: 0.8rem !important;
        left: 0.8rem !important;
        z-index: 99999 !important;
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 4px !important;
    }}

    /* Modern Glassmorphic Sidebar */
    [data-testid="stSidebar"] {{
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }}

    /* Sidebar Inputs Styling */
    [data-testid="stSidebar"] input {{
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #F8FAFC !important;
        border-radius: 8px !important;
    }}
    
    /* Hero Header Styling */
    .hero-title {{
        font-size: 2.3rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        background: linear-gradient(135deg, #FFFFFF 0%, #CBD5E1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }}
    .hero-sub {{
        font-size: 1rem;
        color: #94A3B8;
        margin-bottom: 1.25rem;
        line-height: 1.5;
    }}
    
    /* Connection Status Pill */
    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.2);
        color: #34D399;
        padding: 6px 16px;
        border-radius: 9999px;
        font-size: 0.85em;
        font-weight: 500;
    }}

    /* Centered Admin Card Container */
    .admin-card {{
        background-color: #131C2E;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2.5rem;
        max-width: 540px;
        margin: 2rem auto;
        box-shadow: 0 16px 36px rgba(0, 0, 0, 0.4);
    }}

    /* Messaging App Layout: User on Right, Ashish on Left */
    .user-row {{
        display: flex;
        justify-content: flex-end;
        width: 100%;
        margin-bottom: 1.2rem;
    }}

    .chat-bubble-user {{
        max-width: 80%;
        background: linear-gradient(135deg, #4F46E5 0%, #3730A3 100%);
        color: #FFFFFF;
        padding: 14px 18px;
        border-radius: 18px 18px 4px 18px;
        box-shadow: 0 6px 18px rgba(79, 70, 229, 0.25);
        font-size: 0.95rem;
        line-height: 1.5;
        transition: transform 0.2s ease;
    }}
    .chat-bubble-user:hover {{
        transform: translateY(-2px);
    }}

    .chat-sender-user {{
        font-size: 0.75rem;
        font-weight: 700;
        color: rgba(255, 255, 255, 0.75);
        margin-bottom: 4px;
        text-align: right;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    .assistant-row {{
        display: flex;
        justify-content: flex-start;
        width: 100%;
        margin-bottom: 1.2rem;
    }}

    .chat-bubble-assistant {{
        max-width: 85%;
        background-color: #131C2E;
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #F8FAFC;
        padding: 14px 20px;
        border-radius: 18px 18px 18px 4px;
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.3);
        font-size: 0.95rem;
        line-height: 1.55;
        transition: all 0.2s ease;
    }}
    .chat-bubble-assistant:hover {{
        border-color: rgba(99, 102, 241, 0.35);
        box-shadow: 0 12px 28px rgba(99, 102, 241, 0.15);
        transform: translateY(-2px);
    }}

    .chat-sender-assistant {{
        font-size: 0.78rem;
        font-weight: 700;
        color: #6366F1;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* Interactive Buttons with Hover / Click Effects */
    .stButton > button {{
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%) !important;
        color: #F8FAFC !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.2rem !important;
        font-weight: 500 !important;
        font-size: 0.88em !important;
        transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }}

    .stButton > button:hover {{
        background: linear-gradient(135deg, #4F46E5 0%, #3730A3 100%) !important;
        border-color: #6366F1 !important;
        color: #FFFFFF !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.35) !important;
    }}

    .stButton > button:active {{
        transform: translateY(0) scale(0.98) !important;
        box-shadow: 0 2px 6px rgba(99, 102, 241, 0.2) !important;
    }}

    /* Distinct & Highly Visible Chat Text Area */
    [data-testid="stChatInput"] {{
        border: 1.5px solid #4F46E5 !important;
        border-radius: 14px !important;
        background-color: #0F172A !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
        padding: 4px !important;
    }}

    [data-testid="stChatInput"] textarea {{
        color: #F8FAFC !important;
        font-size: 0.95rem !important;
        background-color: transparent !important;
    }}

    [data-testid="stChatInput"]:focus-within {{
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.3) !important;
    }}

    /* Source Passage Cards */
    .source-box {{
        background-color: #0F172A;
        border-left: 3px solid #6366F1;
        padding: 10px 14px;
        margin: 8px 0;
        border-radius: 6px;
        font-size: 0.88em;
        color: #94A3B8;
        line-height: 1.55;
    }}

    /* Hide Footer only, Keep Header Toggle Controls Active */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    {splash_html}
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
default_mongo_uri = get_secret("MONGODB_URI")
mongo_db_name = get_secret("MONGODB_DB_NAME") or "RAG-bot"
admin_secret_key = get_secret("ADMIN_SECRET_KEY") or "admin123"
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

if "show_admin_login_page" not in st.session_state:
    st.session_state.show_admin_login_page = False

if "is_admin_authenticated" not in st.session_state:
    st.session_state.is_admin_authenticated = False

# Safe Hugging Face Cache Setup
hf_cache_dir = os.path.join(tempfile.gettempdir(), "hf_cache")
os.environ["HF_HOME"] = hf_cache_dir

# Embedding Helper with Fallback support for 429 Rate Limits
def get_embeddings_model(google_key, use_fallback=False):
    if use_fallback:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", cache_folder=hf_cache_dir)
        except Exception:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", cache_folder=hf_cache_dir)
    else:
        return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=google_key)

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
    # Discrete Admin Access Dropdown in Sidebar Corner
    try:
        admin_popover = st.popover("⋮ Admin Settings")
    except Exception:
        admin_popover = st.expander("⋮ Admin Settings")

    with admin_popover:
        st.markdown("**Administrator Portal**")
        if st.session_state.is_admin_authenticated:
            st.success("Admin Authenticated")
            if st.button("Open Ingestion Dashboard", use_container_width=True):
                st.session_state.show_admin_login_page = True
                st.rerun()
            if st.button("Admin Logout", use_container_width=True):
                st.session_state.is_admin_authenticated = False
                st.session_state.show_admin_login_page = False
                st.rerun()
        else:
            st.caption("Click below to open the Admin Login portal.")
            if st.button("Admin Login", use_container_width=True):
                st.session_state.show_admin_login_page = True
                st.rerun()

    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 1.25rem; font-weight: 700; color: #F8FAFC;'>Ashish AI</div>", unsafe_allow_html=True)
    st.caption("IIITD Administrative Officer & Guide")
    
    st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)
    
    # Student API Key Override Section
    st.subheader("API Settings")
    st.caption("Optionally enter your custom API key below or leave blank to use system defaults.")
    
    user_groq_key = st.text_input(
        "Groq API Key",
        value="",
        type="password",
        placeholder="put your api key here",
        help="put your api key here"
    )
    
    user_google_key = st.text_input(
        "Google Gemini API Key",
        value="",
        type="password",
        placeholder="put your api key here",
        help="put your api key here"
    )
    
    # Determine active API keys (User override if typed, otherwise system default)
    groq_api_key_input = user_groq_key.strip() if user_groq_key.strip() else default_groq_key
    google_api_key_input = user_google_key.strip() if user_google_key.strip() else default_google_key

    st.markdown("---")
    
    # LLM Selector
    model_choice = st.selectbox(
        "Select Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
        index=0
    )
    
    # Advanced Model Settings Expander
    with st.expander("Model Parameters"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        k_retrieved = st.slider("Retrieved Chunks (k)", 1, 10, 6, 1)

    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    
    # Sleek Clear Button
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Read-Only Function to load existing vector embeddings from MongoDB
def load_from_mongodb():
    if mongo_coll is None or not google_api_key_input:
        return False
    
    try:
        records = list(mongo_coll.find({}))
        if not records:
            return False
        
        embeddings = get_embeddings_model(google_api_key_input, use_fallback=False)
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
        return False

# Auto-load existing vectors from MongoDB on app launch
if st.session_state.vectors is None and google_api_key_input:
    load_from_mongodb()

# ==============================================================================
# VIEW ROUTING: CENTERED ADMIN LOGIN / INGESTION PAGE vs STUDENT CHAT WORKSPACE
# ==============================================================================
if st.session_state.show_admin_login_page:
    # --------------------------------------------------------------------------
    # CENTERED ADMIN LOGIN PAGE
    # --------------------------------------------------------------------------
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='hero-title' style='text-align: center;'>Administrator Portal</div>", unsafe_allow_html=True)
        st.markdown("<div class='hero-sub' style='text-align: center;'>Manage and Ingest IIITD Vector Embeddings</div>", unsafe_allow_html=True)

        if not st.session_state.is_admin_authenticated:
            st.markdown("<div class='admin-card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='margin-top:0; color:#F8FAFC;'>Admin Authentication</h4>", unsafe_allow_html=True)
            admin_pass_input = st.text_input("Enter Admin Password", type="password", help="Enter password to access document ingestion.")
            
            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                if st.button("Login", use_container_width=True):
                    if admin_pass_input and admin_pass_input == admin_secret_key:
                        st.session_state.is_admin_authenticated = True
                        st.success("Access Granted")
                        st.rerun()
                    else:
                        st.error("Invalid Admin Password!")
            with c_btn2:
                if st.button("Return to Chat", use_container_width=True):
                    st.session_state.show_admin_login_page = False
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='admin-card'>", unsafe_allow_html=True)
            st.success("Admin Session Active")
            st.markdown("### Document Ingestion Panel")
            
            load_assets = st.checkbox("Include ./Assets PDFs", value=True)
            assets_files = []
            if os.path.exists("./Assets"):
                assets_files = [f for f in os.listdir("./Assets") if f.endswith(".pdf")]
            
            uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
            chunk_size = st.number_input("Chunk Size", min_value=100, max_value=4000, value=1000, step=100)
            chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=1000, value=200, step=50)

            def create_and_store_in_mongodb(show_toast=True):
                active_google_key = google_api_key_input
                if not active_google_key:
                    st.error("Google Gemini API Key is required to embed documents!")
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

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                final_documents = text_splitter.split_documents(docs)
                
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

                new_tuples = []
                use_fallback_model = False
                
                if not chunks_to_embed:
                    if show_toast:
                        st.info(f"All {len(cached_tuples)} document chunks are indexed!")
                else:
                    status_box = st.status(f"Ingesting {len(chunks_to_embed)} new chunks into database...", expanded=True)
                    status_box.write(f"Found {len(cached_tuples)} existing chunks in database.")
                    
                    batch_size = 10
                    current_index = 0
                    total_new = len(chunks_to_embed)
                    
                    while current_index < total_new:
                        batch = chunks_to_embed[current_index : current_index + batch_size]
                        batch_texts = [item[1].page_content for item in batch]
                        
                        provider_label = "Hugging Face Fallback" if use_fallback_model else "Google Gemini"
                        status_box.write(f"Processing chunk {current_index + 1}–{min(current_index + len(batch), total_new)} of {total_new} using {provider_label}...")
                        
                        embeddings = get_embeddings_model(active_google_key, use_fallback=use_fallback_model)
                        
                        embedded_batch = False
                        for attempt in range(3):
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
                                    try:
                                        mongo_coll.insert_many(mongo_inserts)
                                    except Exception:
                                        pass
                                
                                current_index += len(batch)
                                embedded_batch = True
                                time.sleep(0.5)
                                break
                            except Exception as e:
                                err_str = str(e)
                                if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str):
                                    if not use_fallback_model:
                                        status_box.warning("Gemini Rate Limit 429 hit. Switching to Hugging Face all-MiniLM-L6-v2...")
                                        status_box.write(f"Resuming embedding directly from chunk {current_index + 1} of {total_new} using Hugging Face...")
                                        use_fallback_model = True
                                        embeddings = get_embeddings_model(active_google_key, use_fallback=True)
                                        time.sleep(1)
                                    else:
                                        status_box.warning(f"Rate limit pause. Retrying batch from chunk {current_index + 1} in 4 seconds...")
                                        time.sleep(4)
                                else:
                                    status_box.warning(f"Provider notice: {e}. Switching to Hugging Face fallback...")
                                    use_fallback_model = True
                                    embeddings = get_embeddings_model(active_google_key, use_fallback=True)
                                    time.sleep(1)
                        
                        if not embedded_batch and not use_fallback_model:
                            use_fallback_model = True

                    status_box.update(label="Ingestion Complete!", state="complete", expanded=False)

                all_tuples = cached_tuples + new_tuples
                text_embeddings = [(t[0], t[1]) for t in all_tuples]
                metadatas = [t[2] for t in all_tuples]
                
                active_embeddings = get_embeddings_model(active_google_key, use_fallback=use_fallback_model)
                vector_store = FAISS.from_embeddings(text_embeddings=text_embeddings, embedding=active_embeddings, metadatas=metadatas)
                
                st.session_state.vectors = vector_store
                st.session_state.doc_count = len(docs)
                st.session_state.chunk_count = len(all_tuples)
                st.session_state.indexed_files = list(set(loaded_file_names))
                
                if show_toast:
                    st.success(f"RAG Index Ready! Loaded {len(all_tuples)} chunks into MongoDB.")
                return True

            if st.button("Embed & Store Data into MongoDB", use_container_width=True):
                create_and_store_in_mongodb(show_toast=True)

            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
            if st.button("Back to Student Chat", use_container_width=True):
                st.session_state.show_admin_login_page = False
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

else:
    # --------------------------------------------------------------------------
    # MAIN STUDENT CHAT INTERFACE
    # --------------------------------------------------------------------------
    st.markdown("<div class='hero-title'>Welcome Students</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Ashish, your Administrative Officer and online guide for IIITD</div>", unsafe_allow_html=True)

    # Connection status pill
    if st.session_state.vectors is not None:
        st.markdown("<div class='status-pill'>Connection to documents successful, ask your queries now</div><br>", unsafe_allow_html=True)

    # Render Chat History in Messaging App Layout (User on Right, Ashish on Left)
    for message in st.session_state.messages:
        role = message["role"]
        if role == "user":
            st.markdown(f"""
                <div class="user-row">
                    <div class="chat-bubble-user">
                        <div class="chat-sender-user">You</div>
                        <div>{message["content"]}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="assistant-row">
                    <div class="chat-bubble-assistant">
                        <div class="chat-sender-assistant">Ashish</div>
                        <div>{message["content"]}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            if "response_time" in message:
                st.caption(f"Sent · {message['response_time']:.2f}s response time")
                
            if "sources" in message and message["sources"]:
                with st.expander(f"View {len(message['sources'])} Source Passages"):
                    for i, doc in enumerate(message["sources"]):
                        source_name = doc.metadata.get("source", "Document")
                        page_num = doc.metadata.get("page", 0) + 1
                        st.markdown(f"**Source {i+1}:** `{os.path.basename(source_name)}` (Page {page_num})")
                        st.markdown(f"<div class='source-box'>{doc.page_content}</div>", unsafe_allow_html=True)

    # Chat Input & RAG Pipeline Execution for Ashish Persona
    user_query = st.chat_input("Send a message to Ashish...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Display user query on the RIGHT side
        st.markdown(f"""
            <div class="user-row">
                <div class="chat-bubble-user">
                    <div class="chat-sender-user">You</div>
                    <div>{user_query}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Process assistant response on the LEFT side
        if not groq_api_key_input:
            st.error("Groq API Key missing!")
        elif st.session_state.vectors is None:
            success = load_from_mongodb()
            if not success:
                st.error("No documents loaded in database.")
                st.stop()
        
        if st.session_state.vectors is not None:
            try:
                with st.spinner("Ashish is typing..."):
                    try:
                        llm = ChatGroq(
                            groq_api_key=groq_api_key_input,
                            model=model_choice,
                            temperature=temperature
                        )
                    except Exception:
                        llm = ChatGroq(
                            groq_api_key=groq_api_key_input,
                            model="mixtral-8x7b-32768",
                            temperature=temperature
                        )

                    # Direct, Professional, Welcoming Administrative Persona Prompt
                    prompt_template = """ 
                    You are Ashish, the Administrative Officer and official student guide for IIITD.
                    Synthesize a direct, clear, professional, yet welcoming response to the student's question.
                    Include direct, factual, and accurate answers grounded in the provided document context passages.
                    Maintain an authoritative yet helpful administrative tone—direct, precise, and practical without being overly casual or verbose.
                    Seamlessly incorporate code snippets, mathematical formulas, algorithms, and textual details from the context.
                    If the context does not contain sufficient information to answer the question, state politely and directly that the IIITD official records do not contain those details. if you don't get the answer please from the context please just say 'I'm not sure I don't have knowledge of that'
                    Absolutely do not answer anything out of context just say 'please ask Questions related to IIITD only'
                    Never give out the name of the document in the answer,
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
                    
                    try:
                        response = llm.invoke(formatted_prompt)
                    except Exception as llm_err:
                        err_str = str(llm_err)
                        if "429" in err_str or "rate_limit" in err_str.lower():
                            st.warning("Groq rate limited. Switching to fallback model mixtral-8x7b-32768...")
                            fallback_llm = ChatGroq(
                                groq_api_key=groq_api_key_input,
                                model="mixtral-8x7b-32768",
                                temperature=temperature
                            )
                            response = fallback_llm.invoke(formatted_prompt)
                        else:
                            raise llm_err

                    elapsed_time = time.time() - start_time
                    answer = response.content

                    # Render Ashish's answer on the LEFT side
                    st.markdown(f"""
                        <div class="assistant-row">
                            <div class="chat-bubble-assistant">
                                <div class="chat-sender-assistant">Ashish</div>
                                <div>{answer}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.caption(f"Sent · {elapsed_time:.2f}s response time")

                    if retrieved_docs:
                        with st.expander(f"View {len(retrieved_docs)} Source Passages"):
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
                st.error(f"Error executing query: {e}")