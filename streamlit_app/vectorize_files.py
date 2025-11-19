"""
Script to vectorize books and save to disk.
Run once per book, or when you update chunking strategy.

Usage:
    python vectorize_books.py --book guerra_y_paz
    python vectorize_books.py --all
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv(override=True)

# Configuration
DATA_DIR = Path("data")
VECTOR_DIR = Path("vectors")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Available books
BOOKS = {
    "guerra_y_paz": {
        "file": "Guerra_y_paz_cleaned.txt",
        "title": "Guerra y Paz",
        "author": "León Tolstói"
    },
    "crimen_y_castigo": {
        "file": "crime_y_castigo_cleaned.txt",
        "title": "Crimen y Castigo",
        "author": "Fiódor Dostoyevski"
    },
    "don_quijote": {
        "file": "don_quijote.txt",
        "title": "Don Quijote de la Mancha",
        "author": "Miguel de Cervantes"
    },
    "la_escalera": {
        "file": "la_escalera.txt",
        "title": "La Escalera",
        "author": "Miguel Á. H. Parralo."
    }

}


def vectorize_book(book_key: str, override: bool = False):
    """Vectorize a single book and save to disk."""
    
    if book_key not in BOOKS:
        raise ValueError(f"Unknown book: {book_key}. Available: {list(BOOKS.keys())}")
    
    book_info = BOOKS[book_key]
    book_path = DATA_DIR / book_info["file"]
    vector_path = VECTOR_DIR / book_key / str(CHUNK_SIZE) / str(CHUNK_OVERLAP)

    # Skip if vector store already exists and override is False
    if vector_path.exists() and any(vector_path.iterdir()) and not override:
        print(f"Vector store for '{book_info['title']}' already exists at {vector_path}. Skipping vectorization.")
        return None
    
    print(f"\n{'='*60}")
    print(f"Vectorizing: {book_info['title']} by {book_info['author']}")
    print(f"{'='*60}\n")
    
    # Check if book file exists
    if not book_path.exists():
        raise FileNotFoundError(f"Book file not found: {book_path}")
    
    # Load book
    print(f"Loading book from {book_path}...")
    loader = TextLoader(str(book_path), encoding="utf-8")
    docs = loader.load()
    print(f"Loaded {len(docs[0].page_content):,} characters")
    
    # Add metadata
    docs[0].metadata.update({
        "book_key": book_key,
        "title": book_info["title"],
        "author": book_info["author"]
    })
    
    # Split into chunks
    print(f"Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks):,} chunks")
    
    # Create embeddings and vector store
    print(f"Creating embeddings and vector store...")
    #Using Google Generative AI Embeddings (free)
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    
    # Create vector store with persistence
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(vector_path),
        collection_name=book_key
    )
    
    print(f"Successfully vectorized '{book_info['title']}'")
    print(f"Vector store saved to: {vector_path}")
    print(f"Total chunks: {len(chunks):,}")
    
    return vector_store


def main():
    parser = argparse.ArgumentParser(description="Vectorize books for RAG system")
    parser.add_argument(
        "--book",
        type=str,
        choices=list(BOOKS.keys()),
        help="Book to vectorize"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Vectorize all books"
    )
    
    args = parser.parse_args()
    
    # Create directories if needed
    VECTOR_DIR.mkdir(exist_ok=True)
    
    if args.all:
        print(f"Vectorizing all {len(BOOKS)} books...")
        for book_key in BOOKS.keys():
            try:
                vectorize_book(book_key)
            except Exception as e:
                print(f"Error vectorizing {book_key}: {e}")
    elif args.book:
        vectorize_book(args.book)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()