import streamlit as st
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

st.set_page_config(
    page_title="Hindi RAG System - Setup Guide",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Hindi Poems and Stories RAG System")
st.info("⚠️ Qdrant Service Required - Please follow setup instructions below")

st.header("System Requirements")
st.markdown("""
- **OpenAI API Key** - Already configured
- **Qdrant Vector Database** - Required (local or cloud)
""")

st.header("Qdrant Setup Options")

tab1, tab2 = st.tabs(["Local Qdrant (Docker)", "Qdrant Cloud"])

with tab1:
    st.subheader("Option 1: Local Qdrant with Docker")
    st.code("""
# Terminal 1: Start Qdrant
docker run -p 6333:6333 -p 6334:6334 \\
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \\
    qdrant/qdrant
    """, language="bash")
    st.info("After starting Qdrant, refresh this page")

with tab2:
    st.subheader("Option 2: Qdrant Cloud")
    st.markdown("""
    1. Sign up at [cloud.qdrant.io](https://cloud.qdrant.io)
    2. Create a cluster
    3. Get your URL and API key
    4. Update `.env` file:
    """)
    st.code("""
# In .env file
QDRANT_URL=your-cluster-url.gcp.qdrant.tech
QDRANT_API_KEY=your-api-key
    """, language="bash")

st.header("Environment Configuration")
st.code("""
# Current .env configuration:
OPENAI_API_KEY=sk-...[key present]
QDRANT_HOST=localhost
QDRANT_PORT=6333
""", language="bash")

st.header("Next Steps")
st.markdown("""
1. Set up Qdrant using one of the options above
2. Verify Qdrant is accessible at `http://localhost:6333` (for local) or your cloud URL
3. Refresh this page
4. Use the sidebar to initialize the system
""")

st.header("Troubleshooting")
st.markdown("""
- If using Docker/Podman: Make sure the service is running
- Check firewall settings if connecting remotely
- Verify API keys are correct
- For local setup: `curl http://localhost:6333` should return a response
""")

# Test if Qdrant is accessible
try:
    from qdrant_setup import QdrantSetup
    qdrant_setup = QdrantSetup()
    
    # Try to connect
    collections = qdrant_setup.get_client().get_collections()
    st.success(f"✅ Qdrant connection successful! Found {len(collections.collections)} collections")
except Exception as e:
    st.warning(f"⚠️ Qdrant not accessible: {str(e)}")