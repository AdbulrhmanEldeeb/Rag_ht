import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
    PDF_DIR = os.path.join(DATA_DIR, 'pdfs')
    EMBEDDINGS_DIR = os.path.join(DATA_DIR, 'embeddings')
    PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

    # Model configurations
    EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'  # Embedding model name or path
    LANGUAGE_MODEL_NAME = 'llama-3.1-70b-versatile'  # Name of the language model for generation
    
    # Embedding and retrieval settings
    CHUNK_SIZE = 1000  # Dimension of embeddings (depends on model)
    CHUNK_OVERLAB = 200  # Batch size for embedding creation
    VECTOR_STORE_TYPE = 'faiss'  # Vector store type (faiss, annoy, etc.)

    # Retrieval settings
    TOP_K_RETRIEVALS = 5  # Number of top relevant chunks to retrieve

    
    
    # Logging
    LOGGING_LEVEL = 'INFO'  # Logging level (DEBUG, INFO, WARNING, ERROR)

    @staticmethod
    def create_dirs():
        """
        Creates necessary directories if they don't exist.
        """
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        os.makedirs(Config.PDF_DIR, exist_ok=True)
        os.makedirs(Config.EMBEDDINGS_DIR, exist_ok=True)
        os.makedirs(Config.PROCESSED_DIR, exist_ok=True)

# Initialize and create directories
Config.create_dirs()
