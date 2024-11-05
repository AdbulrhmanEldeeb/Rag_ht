import streamlit as st
import os
from dotenv import load_dotenv
from embedding import Embed
from enums import StreamlitConfig 
from chat import Chain

# Load environment variables (API keys, tokens, etc.)
load_dotenv()
# you can use only this load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if  HUGGINGFACEHUB_API_TOKEN is None:
    HUGGINGFACEHUB_API_TOKEN=st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    if HUGGINGFACEHUB_API_TOKEN is None :
        print('please add hugging face token to env') 
if  GROQ_API_KEY is None:
    GROQ_API_KEY=st.secrets["GROQ_API_KEY"]
    if GROQ_API_KEY is None : 

        print('please add gropq api key to env') 

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
# os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable certain TensorFlow optimizations
 
# Set up the Streamlit page configuration with a custom logo and title
PAGE_ICON_PATH = StreamlitConfig.PAGE_ICON_PATH.value
if os.path.exists(PAGE_ICON_PATH):
    st.set_page_config(page_icon=PAGE_ICON_PATH, page_title=StreamlitConfig.PAGE_TITLE.value,layout=StreamlitConfig.LAYOUT.value)
else : 
    FALLBACK_ICON_PATH = StreamlitConfig.FALLBACK_ICON_PATH.value
    st.set_page_config(page_icon=FALLBACK_ICON_PATH, page_title=StreamlitConfig.PAGE_TITLE.value,layout=StreamlitConfig.LAYOUT.value)

# Set the app title and sidebar
st.title(StreamlitConfig.PAGE_TITLE.value)
# lama_image = "/workspaces/Llama3.1-rag-pdf-chat/app_data/lama.jpeg"
st.sidebar.image(StreamlitConfig.SIDEBAR_IMAGE_URL.value)
# st.sidebar.header("RAG Project using Llama3.1 and groq API")
st.sidebar.header(StreamlitConfig.HEADER.value)

st.sidebar.markdown(StreamlitConfig.SIDEBAR_MARKDOWN.value)

# User input for asking questions
input_prompt = st.text_input(StreamlitConfig.INPUT_PROMPT_TITLE.value)

# Display the initial status message to inform the user that documents are being processed
info = st.empty()
info.info(StreamlitConfig.INFO_PROCESS_DOCUMENTS.value)

# Call the function to create vector embeddings for the documents
embedder = Embed()
embedder.vector_embedding()

# Once the documents are processed, update the status message
info.info(StreamlitConfig.RETRIVING_ANSWER.value)

# If the user enters a question
if input_prompt:
    chain=Chain()
    chain.process_chain(input_prompt=input_prompt) 
    # clear info after the response is ready 
    info.empty()