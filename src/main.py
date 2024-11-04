import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
from config import Config
from embedding import Embed
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from langchain.memory import ConversationBufferMemory

# memory = ConversationBufferMemory(input_key="input", output_key="answer", memory_key="history")  # Initialize memory

# Load environment variables (API keys, tokens, etc.)
load_dotenv()
# you can use only this load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

if  hf_token is None:
    print('please add hugging face token to env') 
if  groq_api_key is None:
    print('please add gropq api key to env') 

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
# os.environ["GROQ_API_KEY"] = groq_api_key
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable certain TensorFlow optimizations
 
# Set up the Streamlit page configuration with a custom logo and title
my_logo = "/workspaces/Rag_ht/app_data/icon.png"
if os.path.exists(my_logo):
    st.set_page_config(page_icon=my_logo, page_title="Heat Treatment Chatbot")
else : 
    my_logo = r"app_data\icon.png"
    st.set_page_config(page_icon=my_logo, page_title="Heat Treatment Chatbot")

# Set the app title and sidebar
st.title("Heat Treatment Chatbot")
# lama_image = "/workspaces/Llama3.1-rag-pdf-chat/app_data/lama.jpeg"
st.sidebar.image('https://www.win-therm.com.my/wp-content/uploads/2021/02/1-1251ok-before-1024x774.jpg')
# st.sidebar.header("RAG Project using Llama3.1 and groq API")
st.sidebar.header("Chat with Heat Treatment PDFs")
st.sidebar.markdown(
    """
    This custom chatbot retrives data from heat treatment documents.

    General purpose of this app is to help students to get answers from pdfs easily.

    What makes this chatbot better than other chatbots is that it is more focused on the data given to it.

    Example of Questions to ask chatbot:

    ```
    describe Black-heart process for Malleable iron production.
    ```
    ```
    How do elements like magnesium or boron affect graphite morphology in cast iron?
    ```
    ```
    what are properties of Iron Carbide Phase?
    ```
    """
)


# Initialize the language model (LLM) using the Groq API with Llama3.1
llm = ChatGroq(groq_api_key=groq_api_key, model_name=Config.LANGUAGE_MODEL_NAME)

# Define the prompt template for question answering
prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context.If the provided context does not include the answer, \
provide an answer based on your knowledge.
{context}
<context>
Questions: {input}
"""
)



# User input for asking questions
prompt1 = st.text_input("Add your question here and press enter")

# Display the initial status message to inform the user that documents are being processed
info = st.empty()
info.info("Your documents are being processed, please wait a second...⌛")

# Call the function to create vector embeddings for the documents
Embed.vector_embedding() 


# Once the documents are processed, update the status message
info.info("Vector Store DB is ready. Ask any question from your documents.")

# If the user enters a question
if prompt1:
    # Create a chain to retrieve and process relevant documents using the LLM and prompt
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Retrieve the vector store created from the PDF documents
    retriever = st.session_state.vectors.as_retriever()

    # Clear the info box once the processing starts
    info.write("")

    # Create a retrieval chain that combines the document retriever and the language model
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Measure the time taken to generate the response
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt1})
    print("Response time:", time.process_time() - start)

    # Display the response (answer to the user's question)
    st.write(response["answer"])

    # Use an expander to show relevant document chunks for similarity search
    with st.expander("Document Similarity Search"):
        # Display relevant document chunks that were used to generate the response
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")

