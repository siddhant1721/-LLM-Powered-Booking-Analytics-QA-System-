import os
import pandas as pd
import uuid
import chromadb
from chromadb.config import Settings as ChromaSettings

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llama_index.core import Settings, PromptTemplate, Document
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- Utility Functions ---

def load_llm():
    """Initialize the Groq LLM using Llama 3.3."""
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("Groq API key is not set. Please set GROQ_API_KEY.")
    from llama_index.llms.groq import Groq
    return Groq(model="llama-3.3-70b-versatile", api_key=groq_api_key, request_timeout=120.0)

def get_embed_model():
    """Return the embedding model."""
    return HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)

def get_chroma_client():
    """Initialize and return ChromaDB client with persistence."""
    PERSISTENCE_DIR = os.path.join(os.getcwd(), "chroma_db")
    os.makedirs(PERSISTENCE_DIR, exist_ok=True)
    return chromadb.PersistentClient(
        path=PERSISTENCE_DIR,
        settings=ChromaSettings(anonymized_telemetry=False)
    )

def get_or_create_collection(file_name: str):
    """Get existing collection or create a new one for the CSV."""
    client = get_chroma_client()
    collection_name = f"csv_{file_name.replace('.', '_')}"
    try:
        # In Chroma v0.6.0, list_collections returns a list of collection names.
        collections = client.list_collections()
        if collection_name in collections:
            return client.get_collection(collection_name), True
        else:
            return client.create_collection(collection_name), False
    except Exception as e:
        raise Exception(f"Error accessing ChromaDB: {e}")

def process_csv_for_rag(file_path: str, file_name: str):
    """Process CSV, create embeddings, and return a vector store index."""
    df = pd.read_csv(file_path)
    chroma_collection, exists = get_or_create_collection(file_name)
    if exists and chroma_collection.count() > 0:
        # Use existing embeddings
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        embed_model = get_embed_model()
        Settings.embed_model = embed_model
        index = VectorStoreIndex.from_vector_store(vector_store)
    else:
        # Create new embeddings and store them
        text_data = df.to_string(index=False)
        embed_model = get_embed_model()
        Settings.embed_model = embed_model
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        documents = [Document(text=text_data, id_=f"doc_{file_name}")]
        index = VectorStoreIndex.from_documents(
            documents=documents,
            vector_store=vector_store,
            show_progress=True
        )
    return index

# --- API Setup ---

# Define a default CSV path (ensure this file exists in your folder)
DEFAULT_CSV_PATH = "hotel_bookings_sample.csv"

# Process the CSV and create a query engine during startup.
try:
    index = process_csv_for_rag(DEFAULT_CSV_PATH, os.path.basename(DEFAULT_CSV_PATH))
except Exception as e:
    print("Error processing CSV:", e)
    index = None

if index is not None:
    try:
        llm = load_llm()
        Settings.llm = llm
        # Create the query engine (non-streaming mode)
        query_engine = index.as_query_engine(streaming=False)
    except Exception as e:
        print("Error loading LLM:", e)
        query_engine = None
else:
    query_engine = None

# Create the FastAPI app
app = FastAPI()

class AskRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_query(request: AskRequest):
    """
    Ask a question about the CSV data.
    The API returns an answer based on the pre-processed CSV.
    """
    if query_engine is None:
        raise HTTPException(status_code=500, detail="Query engine not initialized.")
    response = query_engine.query(request.query)
    return {"answer": response.response}
