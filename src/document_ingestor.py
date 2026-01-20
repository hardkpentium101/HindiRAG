import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Union
import uuid
from qdrant_client.http import models

class DocumentIngestor:
    def __init__(self, qdrant_client, collection_name: str):
        """
        Initialize document ingestor with Qdrant client and collection name
        """
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.hash_file_path = f"./{collection_name}_document_hashes.json"

    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate SHA256 hash of a file
        """
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read the file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _load_document_hashes(self) -> Dict[str, str]:
        """
        Load previously saved document hashes from file
        """
        if os.path.exists(self.hash_file_path):
            try:
                with open(self.hash_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}

    def _save_document_hashes(self, hashes: Dict[str, str]) -> None:
        """
        Save document hashes to file
        """
        with open(self.hash_file_path, 'w', encoding='utf-8') as f:
            json.dump(hashes, f, ensure_ascii=False, indent=2)

    def _get_changed_documents(self, data_dir: str) -> List[str]:
        """
        Compare current files with previously hashed files to determine which ones have changed
        Returns list of file paths that have changed or are new
        """
        current_hashes = {}
        changed_files = []

        data_path = Path(data_dir)

        # Get all JSON and TXT files in the directory
        all_files = list(data_path.glob("*.json")) + list(data_path.glob("*.txt"))

        # Load previous hashes
        previous_hashes = self._load_document_hashes()

        # Calculate hashes for current files
        for file_path in all_files:
            file_str = str(file_path)
            current_hash = self._calculate_file_hash(file_str)
            current_hashes[file_str] = current_hash

            # Check if file is new or has changed
            if file_str not in previous_hashes or previous_hashes[file_str] != current_hash:
                changed_files.append(file_str)

        # Also check for deleted files (present in previous but not in current)
        deleted_files = [file for file in previous_hashes if file not in current_hashes]

        # Update the hash file with current hashes
        self._save_document_hashes(current_hashes)

        print(f"Detected {len(changed_files)} changed/new files, {len(deleted_files)} deleted files")
        return changed_files
    
    def load_hindi_texts(self, data_dir: str, only_changed: bool = True) -> List[Dict]:
        """
        Load Hindi poems and stories from data directory
        Expected format: JSON files with 'title', 'author', 'text', 'genre' fields
        If only_changed is True, only load documents from files that have changed since last ingestion
        """
        documents = []

        # Determine which files to process
        if only_changed:
            files_to_process = self._get_changed_documents(data_dir)
            if not files_to_process:
                print("No document changes detected. Skipping ingestion.")
                return []
        else:
            # Process all files
            data_path = Path(data_dir)
            all_files = list(data_path.glob("*.json")) + list(data_path.glob("*.txt"))
            files_to_process = [str(f) for f in all_files]

        print(f"Processing {len(files_to_process)} files")

        # Process JSON files
        json_files = [f for f in files_to_process if f.endswith('.json')]
        print(f"Found {len(json_files)} JSON files to process")

        for json_file in json_files:
            print(f"Processing file: {json_file}")
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    # Handle both single document and list of documents
                    if isinstance(data, dict):
                        data = [data]

                    for item in data:
                        doc = {
                            'id': str(uuid.uuid4()),
                            'title': item.get('title', ''),
                            'author': item.get('author', ''),
                            'text': item.get('text', ''),
                            'genre': item.get('genre', 'story'),  # Default to story if not specified
                            'source_file': str(json_file)
                        }
                        documents.append(doc)

                    print(f"  - Loaded {len(data)} documents from {Path(json_file).name}")
            except json.JSONDecodeError as e:
                print(f"  - Error reading {json_file}: {e}")
            except Exception as e:
                print(f"  - Unexpected error reading {json_file}: {e}")

        # Process text files
        txt_files = [f for f in files_to_process if f.endswith('.txt')]
        for txt_file in txt_files:
            print(f"Processing text file: {txt_file}")
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()

                    # Simple splitting for multiple poems/stories in one file
                    # Assuming each poem/story is separated by double newlines
                    texts = text.split('\n\n')

                    for i, t in enumerate(texts):
                        if t.strip():
                            doc = {
                                'id': str(uuid.uuid4()),
                                'title': f"{Path(txt_file).stem}_{i}",
                                'author': 'Unknown',
                                'text': t.strip(),
                                'genre': 'story',  # Default to story for txt files
                                'source_file': str(txt_file)
                            }
                            documents.append(doc)

                    print(f"  - Loaded {len([t for t in texts if t.strip()])} text chunks from {Path(txt_file).name}")
            except Exception as e:
                print(f"  - Error reading {txt_file}: {e}")

        print(f"Total documents loaded: {len(documents)}")
        return documents
    
    def chunk_text(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """
        Split text into chunks of specified size
        """
        # Split by sentences to maintain coherence
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def ingest_documents(self, documents: List[Dict], embedding_function) -> None:
        """
        Ingest documents into Qdrant collection with embeddings
        """
        import time
        from httpx import TimeoutException
        from qdrant_client.http.exceptions import ResponseHandlingException

        points = []

        for idx, doc in enumerate(documents):
            # Chunk the text if it's too long
            text_chunks = self.chunk_text(doc['text'])

            for i, chunk in enumerate(text_chunks):
                # Generate embedding for the chunk
                embedding = embedding_function(chunk)

                # Create a unique ID for this chunk - using UUID for compatibility
                chunk_id = str(uuid.uuid4())

                # Prepare payload with metadata
                payload = {
                    'title': doc['title'],
                    'author': doc['author'],
                    'genre': doc['genre'],
                    'source_file': doc['source_file'],
                    'original_id': doc['id'],
                    'chunk_index': i,
                    'full_text': chunk
                }

                # Add point to the list
                points.append(models.PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload=payload
                ))

            # Batch upload every 50 points to avoid timeout issues (reduced from 100)
            if len(points) >= 50:
                if points:
                    success = False
                    attempts = 0
                    max_attempts = 3
                    while not success and attempts < max_attempts:
                        try:
                            self.qdrant_client.upsert(
                                collection_name=self.collection_name,
                                points=points
                            )
                            print(f"Batch uploaded {len(points)} document chunks to Qdrant collection '{self.collection_name}'")
                            points = []  # Reset points list after uploading
                            success = True
                        except (ResponseHandlingException, TimeoutException) as e:
                            attempts += 1
                            print(f"Upload attempt {attempts} failed: {e}")
                            if attempts < max_attempts:
                                print(f"Retrying in 2 seconds... (attempt {attempts + 1})")
                                time.sleep(2)
                            else:
                                print(f"Max attempts reached. Skipping this batch of {len(points)} points.")
                                points = []  # Clear the problematic points to continue

            # Progress indicator
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(documents)} documents...")

        # Upload remaining points
        if points:
            success = False
            attempts = 0
            max_attempts = 3
            while not success and attempts < max_attempts:
                try:
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    print(f"Ingested {len(points)} final document chunks into Qdrant collection '{self.collection_name}'")
                    success = True
                except (ResponseHandlingException, TimeoutException) as e:
                    attempts += 1
                    print(f"Final upload attempt {attempts} failed: {e}")
                    if attempts < max_attempts:
                        print(f"Retrying in 2 seconds... (attempt {attempts + 1})")
                        time.sleep(2)
                    else:
                        print(f"Max attempts reached for final batch. {len(points)} points not ingested.")
    
    def load_and_ingest(self, data_dir: str, embedding_function, only_changed: bool = True) -> int:
        """
        Load documents from directory and ingest them into Qdrant
        If only_changed is True, only ingest documents from files that have changed since last ingestion
        """
        print(f"Loading documents from {data_dir}")
        documents = self.load_hindi_texts(data_dir, only_changed=only_changed)

        if not documents:
            print("No new or changed documents to ingest.")
            return 0

        print(f"Loaded {len(documents)} documents")

        print("Ingesting documents into Qdrant...")
        self.ingest_documents(documents, embedding_function)

        return len(documents)

# Example usage
if __name__ == "__main__":
    # This would be called from the main application
    pass