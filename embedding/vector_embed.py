from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import streamlit as st 
import time 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import Config 
import os 


# Function to handle vector embedding of documents



class Embed: 
    def __init__(self) -> None:
         pass


    def vector_embedding(self):
        # Check if vectors are already stored in session state
        if "vectors" not in st.session_state:
            start = time.time()  # Track start time

            # Initialize Hugging Face embeddings (pre-trained model)
            st.session_state.embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL_NAME
            )

            # Check if the FAISS index file exists
            vector_store_path = Config.EMBEDDINGS_DIR

            if os.path.exists(vector_store_path):
                # Load FAISS from disk
                st.session_state.vectors = FAISS.load_local(
                    vector_store_path,
                    st.session_state.embeddings,
                    allow_dangerous_deserialization=True,
                )
                st.write("Loaded vector store from disk.")

            else:
                # Load all PDFs from the "/workspaces/Rag_ht/data/pdfs" folder
                st.session_state.loader = PyPDFDirectoryLoader(Config.PDF_DIR)
                st.session_state.docs = st.session_state.loader.load()

                # Split the loaded documents into chunks
                st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAB
                )
                st.session_state.final_documents = (
                    st.session_state.text_splitter.split_documents(st.session_state.docs)
                )

                # Create vector embeddings for the split documents
                st.session_state.vectors = FAISS.from_documents(
                    st.session_state.final_documents, st.session_state.embeddings
                )

                # Save FAISS index to disk
                st.session_state.vectors.save_local(vector_store_path)
                st.write("Vector store saved to disk.")

            # Display the total processing time
            end = time.time()
            total_time = end - start
            st.write(f"Total time to process documents: {round(total_time/60, 2)} minutes.")
