import streamlit as st
import os
from pathlib import Path
import sys

# Add src directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

# Import modules after adding to path
from rag_system import HindiRAGSystem
from qdrant_setup import QdrantSetup
from document_ingestor import DocumentIngestor
from embedding_generator import get_embedding_function

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None

if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False

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

# Sidebar for configuration and data loading
with st.sidebar:
    st.header("Configuration")

    # Qdrant connection settings
    st.subheader("Qdrant Settings")
    qdrant_host = st.text_input("Qdrant Host", value="localhost")
    qdrant_port = st.number_input("Qdrant Port", value=6333, min_value=1, max_value=65535)

    # Data loading section
    st.subheader("Load Data")
    data_directory = st.text_input("Data Directory Path", value="../data")

    if st.button("Initialize Qdrant Collection"):
        try:
            qdrant_setup = QdrantSetup(host=qdrant_host, port=qdrant_port)
            qdrant_setup.create_collection()
            st.success("Qdrant collection initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing Qdrant: {str(e)}")

    if st.button("Load Documents to Qdrant") and os.path.exists(data_directory):
        try:
            # Initialize RAG system components
            qdrant_setup = QdrantSetup(host=qdrant_host, port=qdrant_port)
            qdrant_client = qdrant_setup.get_client()
            collection_name = qdrant_setup.get_collection_name()

            # Create document ingestor
            ingestor = DocumentIngestor(qdrant_client, collection_name)

            # Get embedding function
            embedding_func = get_embedding_function()

            # Load and ingest documents
            num_docs = ingestor.load_and_ingest(data_directory, embedding_func)

            st.session_state.documents_loaded = True
            st.success(f"Successfully loaded {num_docs} documents into Qdrant!")
        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")

    # Initialize RAG system
    if st.button("Initialize RAG System"):
        try:
            st.session_state.rag_system = HindiRAGSystem()
            st.success("RAG System initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing RAG system: {str(e)}")

# Main content area
st.header("Ask Questions About Hindi Literature")

# Check if RAG system is initialized
if st.session_state.rag_system is None:
    st.warning("Please initialize the RAG system from the sidebar before asking questions.")
else:
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
                        st.write("**Genre:**", doc.get('genre', 'N/A'))
                        st.write("**Source:**", doc.get('source_file', 'N/A'))
                        st.write("**Text:**", doc.get('text', 'N/A'))
                    st.write(doc)
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")

# Information section
st.divider()
st.markdown("""
### How to Use:
1. Enter your Qdrant server details in the sidebar
2. Place your Hindi poems/stories in the data directory (JSON or TXT format)
3. Click "Initialize Qdrant Collection" to create the collection
4. Click "Load Documents to Qdrant" to ingest your data
5. Click "Initialize RAG System" to prepare for queries
6. Ask questions about your Hindi literature in the main panel

### Data Format:
- JSON files should contain objects with: title, author, text, genre
- TXT files will be processed as continuous text
""")

# Show status
st.sidebar.divider()
st.sidebar.markdown("**Status:**")
st.sidebar.write(f"- Qdrant Collection: {'✅ Ready' if st.session_state.documents_loaded else '❌ Not loaded'}")
st.sidebar.write(f"- RAG System: {'✅ Initialized' if st.session_state.rag_system else '❌ Not initialized'}")