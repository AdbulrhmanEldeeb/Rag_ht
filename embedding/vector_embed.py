import os
import time
import streamlit as st
from configuration import Config
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Embed:
    def __init__(self):
        # Initialize instance attributes for embeddings and vectors
        self.vector_store_path = Config.EMBEDDINGS_DIR
        self.pdf_dir = Config.PDF_DIR
        self.chunk_size = Config.CHUNK_SIZE
        self.chunk_overlap = Config.CHUNK_OVERLAB
        self.model_name = Config.EMBEDDING_MODEL_NAME
        self.vectors = None
        self.embeddings = None
        self.loader = None
        self.text_splitter = None
        self.final_documents = None

    def load_embeddings(self):
        """Initialize Hugging Face embeddings using the pre-trained model name."""
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        st.session_state.embeddings = self.embeddings

    def load_faiss_from_disk(self):
        """Load FAISS index from local storage if available."""
        self.vectors = FAISS.load_local(
            self.vector_store_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        st.session_state.vectors = self.vectors
        st.write("Loaded vector store from disk.")

    def create_faiss_from_documents(self):
        """Load documents, split them, create embeddings, and save FAISS index."""
        # Load PDFs and split documents
        self.loader = PyPDFDirectoryLoader(self.pdf_dir)
        docs = self.loader.load()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        self.final_documents = self.text_splitter.split_documents(docs)
        
        # Generate vector embeddings and save them
        self.vectors = FAISS.from_documents(self.final_documents, self.embeddings)
        self.vectors.save_local(self.vector_store_path)
        st.session_state.vectors = self.vectors
        st.write("Vector store saved to disk.")

    def vector_embedding(self):
        """Process documents and create vector embeddings if not in session state."""
        if "vectors" not in st.session_state:
            start = time.time()  # Start time for processing

            # Initialize embeddings if not already loaded
            if not self.embeddings:
                self.load_embeddings()

            # Check if vector store exists on disk, load or create as needed
            if os.path.exists(self.vector_store_path):
                self.load_faiss_from_disk()
            else:
                self.create_faiss_from_documents()

            # Display total processing time
            total_time = time.time() - start
            st.write(f"Total time to process documents: {round(total_time/60, 2)} minutes.")

# Usage
# embedder = Embed()
# embedder.vector_embedding()
