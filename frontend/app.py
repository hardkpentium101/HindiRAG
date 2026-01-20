import streamlit as st
import os
from pathlib import Path
import sys

# Add src directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from rag_system import HindiRAGSystem
from qdrant_setup import QdrantSetup
from document_ingestor import DocumentIngestor
from embedding_generator import get_embedding_function

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
    st.session_state.system_initialized = False
    st.session_state.current_model = os.getenv("LLM_PROVIDER", "huggingface")
    st.session_state.model_configured = False

# Model selection in sidebar
with st.sidebar:
    st.header("Model Selection")

    # Define available models
    available_models = {
        "Hugging Face (Open Source)": "huggingface",
        "OpenAI GPT": "openai",
        "Local Model": "local",
        "Anthropic Claude": "anthropic",
        "Google Gemini": "google",
        "Ollama": "ollama"
    }

    # Model selection dropdown
    selected_model_display = st.selectbox(
        "Choose AI Model:",
        options=list(available_models.keys()),
        index=list(available_models.values()).index(st.session_state.current_model) if st.session_state.current_model in available_models.values() else 0
    )

    selected_model = available_models[selected_model_display]

    # Update session state if model changed
    if selected_model != st.session_state.current_model:
        st.session_state.current_model = selected_model
        st.session_state.model_configured = False  # Reset model configuration flag
        if st.session_state.rag_system is not None:
            st.session_state.rag_system = None  # Clear existing system to force reinitialization
            st.session_state.system_initialized = False

# Initialize the RAG system with selected model
if not st.session_state.system_initialized or not st.session_state.model_configured:
    # Initialize the RAG system automatically
    try:
        # Use HuggingFace as default if no model is selected or if there are issues with the selected model
        provider_to_use = st.session_state.current_model
        if provider_to_use == "openai" and not os.getenv("OPENAI_API_KEY"):
            # If OpenAI is selected but no API key is available, default to HuggingFace
            provider_to_use = "huggingface"
            st.warning("OpenAI API key not found. Using HuggingFace as default.")

        st.session_state.rag_system = HindiRAGSystem(llm_provider=provider_to_use)
        st.session_state.system_initialized = True
        st.session_state.model_configured = True
    except Exception as e:
        st.session_state.rag_system = None
        st.session_state.system_initialized = False
        st.session_state.model_configured = False
        st.error(f"Error initializing system with {selected_model_display}: {str(e)}")
        # Try to initialize with HuggingFace as fallback
        try:
            st.session_state.rag_system = HindiRAGSystem(llm_provider="huggingface")
            st.session_state.system_initialized = True
            st.session_state.model_configured = True
            st.success("System initialized with HuggingFace as fallback.")
        except Exception as fallback_error:
            st.error(f"Fallback initialization also failed: {str(fallback_error)}")

if 'documents_loaded' not in st.session_state:
    # Automatically load documents if system is initialized
    if st.session_state.get('system_initialized', False):
        try:
            # Initialize RAG system components
            qdrant_setup = QdrantSetup()
            qdrant_client = qdrant_setup.get_client()
            collection_name = qdrant_setup.get_collection_name()

            # Create document ingestor
            ingestor = DocumentIngestor(qdrant_client, collection_name)

            # Get embedding function
            embedding_func = get_embedding_function()

            # Load and ingest documents from the data directory
            data_directory = "./data"
            if os.path.exists(data_directory):
                num_docs = ingestor.load_and_ingest(data_directory, embedding_func)
                st.session_state.documents_loaded = True
                st.session_state.num_documents = num_docs
            else:
                st.session_state.documents_loaded = False
                st.session_state.num_documents = 0
        except Exception as e:
            st.session_state.documents_loaded = False
            st.session_state.num_documents = 0
            st.error(f"Error loading documents: {str(e)}")
    else:
        st.session_state.documents_loaded = False
        st.session_state.num_documents = 0

# Streamlit app configuration
st.set_page_config(
    page_title="Hindi RAG System",
    page_icon="📚",
    layout="wide"
)

# Title and description
st.title("📚 Hindi Poems and Stories RAG System")
st.markdown("""
This application allows you to search and ask questions about Hindi poems and stories using advanced RAG technology.
""")

# Sidebar for configuration and status
with st.sidebar:
    st.header("System Status")

    # Show initialization status
    if st.session_state.get('system_initialized', False):
        st.success("✅ RAG System: Initialized")
    else:
        st.error("❌ RAG System: Not initialized")

    if st.session_state.get('documents_loaded', False):
        num_docs = st.session_state.get('num_documents', 0)
        st.success(f"✅ Documents: Loaded ({num_docs} documents)")
    else:
        st.warning("⚠️ Documents: Not loaded")

    st.divider()

    st.header("Configuration")

    # Show current Qdrant configuration
    st.subheader("Qdrant Configuration")
    st.info(f"Using environment config:\nURL: {os.getenv('QDRANT_URL', 'Not set')}")

    # Show current LLM configuration
    st.subheader("LLM Configuration")
    st.info(f"Current provider: {st.session_state.current_model}")

    # Data directory info
    st.subheader("Data Directory")
    data_directory = "./data"
    if os.path.exists(data_directory):
        st.success(f"✅ Data directory exists: {data_directory}")
        files = os.listdir(data_directory)
        st.write(f"Files: {len(files)} found")
    else:
        st.error(f"❌ Data directory missing: {data_directory}")

# Main content area
st.header("Ask Questions About Hindi Literature")

# Check if RAG system is initialized
if st.session_state.rag_system is None:
    st.error("❌ RAG system failed to initialize. Check the configuration and logs.")
else:
    # Show system ready message
    if st.session_state.get('system_initialized', False):
        st.success("✅ System ready! You can ask questions below.")

        # Question input
        question = st.text_area("Enter your question in Hindi or English:", height=100)

        # Search parameters
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Number of relevant documents to retrieve", 1, 10, 5)
        with col2:
            search_button = st.button("Search", type="primary")

        # Process query when button is clicked
        if search_button and question.strip():
            with st.spinner("Searching and generating answer..."):
                try:
                    result = st.session_state.rag_system.query(question, top_k=top_k)

                    # Display answer
                    st.subheader("Answer")
                    st.write(result['answer'])

                    # Display relevant documents
                    st.subheader("Relevant Documents Found")
                    for i, doc in enumerate(result['relevant_documents']):
                        with st.expander(f"Document {i+1}: {doc['title']} by {doc['author']} (Score: {doc['score']:.3f})"):
                            st.write("**Genre:**", doc['genre'])
                            st.write("**Source:**", doc['source_file'])
                            st.write("**Text:**", doc['text'])

                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
    else:
        st.error("❌ System initialization failed. Please check configuration.")

# Information section
st.divider()
st.markdown("""
### How to Use:
1. The system initializes automatically when the app starts
2. Place your Hindi poems/stories in the data directory (JSON or TXT format)
3. The system automatically loads documents from the data directory
4. Ask questions about your Hindi literature in the main panel below

### Data Format:
- JSON files should contain objects with: title, author, text, genre
- TXT files will be processed as continuous text
""")

# Show status
st.sidebar.divider()
st.sidebar.markdown("**Status:**")
st.sidebar.write(f"- Qdrant Collection: {'✅ Ready' if st.session_state.documents_loaded else '❌ Not loaded'}")
st.sidebar.write(f"- RAG System: {'✅ Initialized' if st.session_state.rag_system else '❌ Not initialized'}")