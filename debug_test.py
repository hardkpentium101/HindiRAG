import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_data_loading():
    """Test the data loading functionality"""
    print("Testing data loading functionality...")
    
    try:
        from qdrant_setup import QdrantSetup
        from document_ingestor import DocumentIngestor
        from embedding_generator import get_embedding_function
        
        print("✓ Modules imported successfully")
        
        # Initialize Qdrant
        qdrant_setup = QdrantSetup()
        qdrant_client = qdrant_setup.get_client()
        collection_name = qdrant_setup.get_collection_name()
        
        print(f"✓ Connected to Qdrant, collection: {collection_name}")
        
        # Test embedding function
        embedding_func = get_embedding_function()
        test_embedding = embedding_func("test")
        print(f"✓ Embedding function works, output length: {len(test_embedding)}")
        
        # Test document loading
        data_dir = "./data"
        if os.path.exists(data_dir):
            ingestor = DocumentIngestor(qdrant_client, collection_name)
            num_docs = ingestor.load_and_ingest(data_dir, embedding_func)
            print(f"✓ Successfully loaded {num_docs} documents from {data_dir}")
        else:
            print(f"⚠ Data directory {data_dir} does not exist")
        
        print("\nData loading test completed successfully!")
        
    except Exception as e:
        print(f"✗ Error in data loading: {str(e)}")
        import traceback
        traceback.print_exc()

def test_search_functionality():
    """Test the search functionality"""
    print("\nTesting search functionality...")
    
    try:
        from rag_system import HindiRAGSystem
        
        # Initialize RAG system
        rag_system = HindiRAGSystem()
        print("✓ RAG system initialized")
        
        # Test a simple query
        question = "What is this collection about?"
        result = rag_system.query(question, top_k=2)
        
        print(f"✓ Search completed successfully")
        print(f"Answer length: {len(result['answer']) if result['answer'] else 0} chars")
        print(f"Found {len(result['relevant_documents'])} relevant documents")
        
        print("\nSearch functionality test completed successfully!")
        
    except Exception as e:
        print(f"✗ Error in search functionality: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_loading()
    test_search_functionality()