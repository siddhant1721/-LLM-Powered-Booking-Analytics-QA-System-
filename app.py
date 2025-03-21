import os
import tempfile
import uuid
import pandas as pd
import chromadb
from chromadb.config import Settings as ChromaSettings

from llama_index.core import Settings, PromptTemplate, Document
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import streamlit as st

# Ensure that GROQ_API_KEY is set in your environment
if not os.environ.get("GROQ_API_KEY"):
    raise ValueError("Groq API key is not set. Please set the GROQ_API_KEY environment variable.")

if "id" not in st.session_state:
    st.session_state.id = str(uuid.uuid4())
    st.session_state.file_cache = {}

# Setup persistent directory for ChromaDB
PERSISTENCE_DIR = os.path.join(os.getcwd(), "chroma_db")
os.makedirs(PERSISTENCE_DIR, exist_ok=True)

session_id = st.session_state.id
query_engine = None

@st.cache_resource
def load_llm():
    """
    Initialize the Groq LLM using Llama 3.3.
    Ensure you have set your Groq API key in the environment variable 'GROQ_API_KEY'.
    """
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("Groq API key is not set. Please set GROQ_API_KEY.")
    
    from llama_index.llms.groq import Groq
    return Groq(model="llama-3.3-70b-versatile", api_key=groq_api_key, request_timeout=120.0)

@st.cache_resource
def get_embed_model():
    """Get and cache the embedding model"""
    return HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)

def get_chroma_client():
    """Initialize and return ChromaDB client with persistence"""
    return chromadb.PersistentClient(
        path=PERSISTENCE_DIR,
        settings=ChromaSettings(anonymized_telemetry=False)
    )

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None

def display_csv(file):
    st.markdown("### CSV Preview")
    df = pd.read_csv(file)
    st.dataframe(df)

def get_or_create_collection(file_name):
    """Get existing collection or create a new one"""
    client = get_chroma_client()
    collection_name = f"csv_{file_name.replace('.', '_')}"
    
    # Check if collection exists
    try:
        collections = client.list_collections()
        collection_names = [col.name for col in collections]
        
        if collection_name in collection_names:
            return client.get_collection(collection_name), True
        else:
            return client.create_collection(collection_name), False
    except Exception as e:
        st.error(f"Error accessing ChromaDB: {e}")
        return client.create_collection(collection_name), False

def process_csv_for_rag(file_path, file_name):
    """Process CSV and store in ChromaDB if not already present"""
    # Load CSV data
    df = pd.read_csv(file_path)
    
    # Check if collection exists
    chroma_collection, exists = get_or_create_collection(file_name)
    
    # If collection exists and has data, use it
    if exists and chroma_collection.count() > 0:
        st.success("Using existing embeddings from database")
        # Set up the vector store with the existing collection
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Create the index from the vector store
        embed_model = get_embed_model()
        Settings.embed_model = embed_model
        index = VectorStoreIndex.from_vector_store(vector_store)
    else:
        # Process and store new embeddings
        with st.spinner("Creating embeddings and storing in database..."):
            text_data = df.to_string(index=False)
            
            # Initialize embedding model
            embed_model = get_embed_model()
            Settings.embed_model = embed_model
            
            # Create vector store
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            # Create embeddings and store in ChromaDB
            documents = [Document(text=text_data, id_=f"doc_{file_name}")]
            index = VectorStoreIndex.from_documents(
                documents=documents,
                vector_store=vector_store,
                show_progress=True
            )
    
    return index

with st.sidebar:
    st.header("Add your documents!")
    uploaded_file = st.file_uploader("Choose your `.csv` file", type=["csv"])

    if uploaded_file:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                file_path = temp_file.name

            file_key = f"{session_id}-{uploaded_file.name}"
            
            if file_key not in st.session_state.file_cache:
                # Process CSV and get index
                index = process_csv_for_rag(file_path, uploaded_file.name)
                
                # Initialize LLM
                llm = load_llm()
                Settings.llm = llm
                
                # Create query engine
                query_engine = index.as_query_engine(streaming=True)
                
                # Customize prompt template
                qa_prompt_tmpl = PromptTemplate(
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context above, answer the query in a precise manner. "
                    "If you don't know, say 'I don't know!'.\n"
                    "Query: {query_str}\n"
                    "Answer: "
                )
                query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})
                
                # Cache the query engine
                st.session_state.file_cache[file_key] = query_engine
            else:
                query_engine = st.session_state.file_cache[file_key]

            st.success("Ready to Chat!")
            display_csv(uploaded_file)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

col1, col2 = st.columns([6, 1])

with col1:
    st.header("RAG over CSV using Groq Model")
with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input only if query_engine is available
if "file_cache" in st.session_state and st.session_state.file_cache:
    query_engine = list(st.session_state.file_cache.values())[0]  # Get the first query engine
    
    if prompt := st.chat_input("Ask me anything about your CSV data!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Stream response
            streaming_response = query_engine.query(prompt)
            for chunk in streaming_response.response_gen:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.chat_input("Upload a CSV file to start chatting!", disabled=True)