import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from qdrant_setup import QdrantSetup
from document_ingestor import DocumentIngestor
from embedding_generator import get_embedding_function
from rag_system import HindiRAGSystem

def setup_and_run():
    """
    Main function to setup and run the Hindi RAG system
    """
    print("Setting up Hindi RAG System...")
    
    # Step 1: Initialize Qdrant
    print("\n1. Setting up Qdrant...")
    qdrant_setup = QdrantSetup()
    qdrant_setup.create_collection()
    
    # Step 2: Load and ingest documents
    print("\n2. Loading and ingesting documents...")
    qdrant_client = qdrant_setup.get_client()
    collection_name = qdrant_setup.get_collection_name()
    
    ingestor = DocumentIngestor(qdrant_client, collection_name)
    embedding_func = get_embedding_function()
    
    # Use the data directory
    data_dir = "./data"
    if os.path.exists(data_dir):
        num_docs = ingestor.load_and_ingest(data_dir, embedding_func)
        print(f"   Loaded {num_docs} documents into Qdrant")
    else:
        print(f"   Warning: Data directory '{data_dir}' does not exist")
    
    # Step 3: Initialize RAG system
    print("\n3. Initializing RAG system...")
    rag_system = HindiRAGSystem()
    
    print("\n4. Hindi RAG System is ready!")
    print("\nTo run the Streamlit frontend, execute: streamlit run frontend/app.py")
    
    # Example query
    print("\n5. Testing with a sample query...")
    question = "प्रकृति का वर्णन हिंदी कविता में कैसे किया गया है?"
    result = rag_system.query(question, top_k=3)
    
    print(f"\nQuestion: {question}")
    print(f"Answer: {result['answer']}")
    
    print("\nRelevant documents found:")
    for i, doc in enumerate(result['relevant_documents']):
        print(f"  {i+1}. {doc['title']} by {doc['author']} (Score: {doc['score']:.3f})")

if __name__ == "__main__":
    setup_and_run()