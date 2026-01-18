import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from qdrant_setup import QdrantSetup
from document_ingestor import DocumentIngestor
from embedding_generator import get_embedding_function
from rag_system import HindiRAGSystem

def test_document_retrieval():
    """
    Test function to verify document retrieval is working properly
    """
    print("Testing Hindi RAG System - Document Retrieval Only...")

    # Initialize RAG system
    print("\nInitializing RAG system...")
    rag_system = HindiRAGSystem()

    # Test query
    question = "प्रकृति का वर्णन हिंदी कविता में कैसे किया गया है?"
    
    print(f"\nQuery: {question}")
    
    # Retrieve relevant documents only (without generating answer)
    relevant_docs = rag_system.retrieve_relevant_documents(question, top_k=3)

    print(f"\nFound {len(relevant_docs)} relevant documents:")
    for i, doc in enumerate(relevant_docs):
        print(f"\n  {i+1}. Title: {doc['title']}")
        print(f"     Author: {doc['author']}")
        print(f"     Genre: {doc['genre']}")
        print(f"     Score: {doc['score']:.3f}")
        print(f"     Text Preview: {doc['text'][:100]}...")
        print(f"     Source: {doc['source_file']}")

    print("\nDocument retrieval test completed successfully!")

if __name__ == "__main__":
    test_document_retrieval()