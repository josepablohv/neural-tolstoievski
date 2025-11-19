"""
Streamlit app for Spanish literature Q&A.
Loads pre-built vector stores and allows querying.

Usage:
    streamlit run app.py
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(override=True)

# Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
VECTOR_DIR = Path("vectors")
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


@st.cache_resource
def load_vector_store(book_key: str, chunks, overlap):
    """Load a pre-built vector store from disk."""
    vector_path = VECTOR_DIR / book_key / str(chunks) / str(overlap)
    
    if not vector_path.exists():
        raise FileNotFoundError(
            f"Vector store not found for '{book_key}'. "
            f"Run: python vectorize_books.py --book {book_key}"
        )
    
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    vector_store = Chroma(
        persist_directory=str(vector_path),
        embedding_function=embeddings,
        collection_name=book_key
    )
    
    return vector_store


@st.cache_resource
def load_model():
    """Load the LLM model."""
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


def create_retrieval_tool(vector_store, book_title: str):
    """Create a retrieval tool for the given vector store."""
    tool_description = (
        f"Retrieve relevant passages from '{book_title}' to help answer a question "
        f"by searching the vector store for the text of the book."
    )
    
    @tool(description=tool_description)
    def retrieve_context(query: str) -> str:
        f"""Retrieve relevant passages from '{book_title}' to help answer a question."""
        docs = vector_store.similarity_search(query, k=10)
        
        # Format retrieved documents
        context = "\n\n---\n\n".join([
            f"[Chunk {i+1}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ])
        
        return context
    
    return retrieve_context



def create_book_agent(model, vector_store, book_info):
    """Create a modern LangChain agent with retrieval."""

    retrieval_tool = create_retrieval_tool(vector_store, book_info["title"])

    system_prompt = f"""
Eres un asistente experto en literatura española especializado en '{book_info['title']}' de {book_info['author']}'.

Tienes acceso a una herramienta que recupera pasajes relevantes del libro. Úsala cuando te pregunten preguntas específicas.

Instrucciones:
- Responde en español si la pregunta está en español.
- Cita las fuentes cuando uses la herramienta de recuperación.
- Si no encuentras información relevante en las fuentes, di que no lo sabes.
- Al final de tu respuesta, incluye un breve resumen empezando con 'En resumen...'
"""

    agent = create_agent(
        model=model, 
        tools=[retrieval_tool], 
        system_prompt=system_prompt
    )

    return agent


def main():
    st.set_page_config(
        page_title="Preguntas de Literatura en Español",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Preguntas de Literatura en Español")
    st.markdown("Pregunta sobre libros en español. Powered by RAG y Google Gemini.")
    
    # Sidebar - Book selection
    with st.sidebar:
        st.header("📖 Select Book")
        
        # Check which books are available
        available_books = {
            k: v for k, v in BOOKS.items()
            if (VECTOR_DIR / k).exists()
        }
        
        if not available_books:
            st.error("No books vectorized yet!")
            st.info("Run: `python vectorize_books.py --all`")
            st.stop()
        
        # Book selector
        book_key = st.selectbox(
            "Choose a book/Elige un libro:",
            options=list(available_books.keys()),
            format_func=lambda k: f"{available_books[k]['title']} - {available_books[k]['author']}"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            f"**{available_books[book_key]['title']}**\n\n"
            f"*by {available_books[book_key]['author']}*"
        )
        
        st.markdown("---")
        st.markdown("### Example Questions")
        st.markdown("""
        - ¿Quién es Napoleón?
        - ¿Cuáles son los personajes principales?
        - ¿De qué trata el libro?
        """)
    
    # Load resources
    try:
        vector_store = load_vector_store(book_key, CHUNK_SIZE, CHUNK_OVERLAP)
        model = load_model()
        agent = create_book_agent(model, vector_store, available_books[book_key])
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.exception(e)
        st.stop()
    
    # Main interface
    st.markdown(f"### Haz preguntas sobre *{available_books[book_key]['title']}*")
    
    question = st.text_input(
        "Tu pregunta:",
        placeholder="¿Quién es el protagonista?",
        key="question_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        ask_button = st.button("Ask", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("Clear", use_container_width=True)
    
    if clear_button:
        st.rerun()
    
    if ask_button and question:
        with st.spinner("Searching and generating answer..."):
            start_time = time.time()
            
            try:
                # Get response
                response = agent.invoke({"messages": [{"role": "user", "content": question}]})

                elapsed_time = time.time() - start_time
                
                # Display answer
                st.markdown("###  Answer")
                st.markdown(response["messages"][-1].content)
                
                # Display metrics
                st.caption(f"Response time: {elapsed_time:.2f}s")
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)
    
    elif ask_button and not question:
        st.warning("Please enter a question.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with LangChain, ChromaDB, and Google Gemini"
        " - Jose Pablo Hernandez with a little help from LLMs"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()