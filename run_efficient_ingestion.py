#!/usr/bin/env python3
"""
Script to efficiently manage document ingestion for the Hindi RAG system
Handles large datasets separately to avoid timeout issues
"""

import os
import sys
import json
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from qdrant_setup import QdrantSetup
from document_ingestor import DocumentIngestor
from embedding_generator import get_embedding_function

def main():
    print("Setting up Hindi RAG System with efficient document ingestion...")

    # Step 1: Initialize Qdrant
    print("\n1. Setting up Qdrant...")
    qdrant_setup = QdrantSetup()
    qdrant_client = qdrant_setup.get_client()
    collection_name = qdrant_setup.get_collection_name()

    # Step 2: Get embedding function
    print("\n2. Getting embedding function...")
    embedding_func = get_embedding_function()

    # Step 3: Separate data files into small and large sets
    data_dir = "./data"
    data_path = Path(data_dir)
    all_json_files = list(data_path.glob("*.json"))
    
    # Separate files: small files first, large file last
    small_files = []
    large_files = []
    
    for json_file in all_json_files:
        file_size = json_file.stat().st_size
        if file_size > 100000:  # If file is larger than ~100KB, treat as large
            large_files.append(json_file)
        else:
            small_files.append(json_file)
    
    print(f"Found {len(small_files)} small files and {len(large_files)} large files")
    
    # Step 4: Process small files first (these are the essential documents)
    print("\n3. Processing essential (small) documents first...")
    essential_ingestor = DocumentIngestor(qdrant_client, collection_name)
    
    for json_file in small_files:
        print(f"  Processing: {json_file.name}")
        # Create a temporary directory with just this file
        temp_data_dir = f"./temp_data_{json_file.stem}"
        os.makedirs(temp_data_dir, exist_ok=True)
        
        # Copy the file to temp directory
        import shutil
        shutil.copy2(json_file, temp_data_dir)
        
        # Load and ingest only this file
        documents = essential_ingestor.load_hindi_texts(temp_data_dir)
        if documents:
            essential_ingestor.ingest_documents(documents, embedding_func)
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_data_dir)
    
    print(f"   Loaded {sum(1 for f in small_files)} essential document files")
    
    # Step 5: Process large files separately with extra caution
    if large_files:
        print(f"\n4. Processing large files: {', '.join([f.name for f in large_files])}")
        large_ingestor = DocumentIngestor(qdrant_client, collection_name)
        
        for json_file in large_files:
            print(f"  Processing large file: {json_file.name} (size: {json_file.stat().st_size:,} bytes)")
            
            # Create a temporary directory with just this large file
            temp_data_dir = f"./temp_large_data_{json_file.stem}"
            os.makedirs(temp_data_dir, exist_ok=True)
            
            # Copy the file to temp directory
            import shutil
            shutil.copy2(json_file, temp_data_dir)
            
            # Load and ingest only this file
            documents = large_ingestor.load_hindi_texts(temp_data_dir)
            if documents:
                large_ingestor.ingest_documents(documents, embedding_func)
            
            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_data_dir)
        
        print(f"   Loaded {sum(1 for f in large_files)} large document files")
    
    # Step 6: Initialize RAG system
    print("\n5. Initializing RAG system...")
    from rag_system import HindiRAGSystem
    rag_system = HindiRAGSystem()
    
    print("\n6. Hindi RAG System is ready!")
    print("\nTo run the Streamlit frontend, execute: streamlit run frontend/app.py")

    # Example query
    print("\n7. Testing with a sample query...")
    question = "हिंदी साहित्य क्या है?"
    result = rag_system.query(question, top_k=3)

    print(f"\nQuestion: {question}")
    print(f"Answer: {result['answer']}")

    print("\nRelevant documents found:")
    for i, doc in enumerate(result['relevant_documents']):
        print(f"  {i+1}. {doc['title']} by {doc['author']} (Score: {doc['score']:.3f})")

if __name__ == "__main__":
    main()