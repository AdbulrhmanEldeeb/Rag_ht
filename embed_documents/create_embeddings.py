"""Run This script if you loaded new data in pdf folder and you want to create faiss embedding from it"""
import sys 
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embedding import Embed 

# from configuration  import Config 
embed=Embed()
embed.load_embeddings() 
embed.create_faiss_from_documents()


