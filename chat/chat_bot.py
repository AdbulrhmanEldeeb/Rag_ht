from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os 
from configuration import Config
from enums import Prompts 
import streamlit as st 
import time 
load_dotenv() 
# hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# if  hf_token is None:
#     print('please add hugging face token to env') 
if  GROQ_API_KEY is None:
    print('please add gropq api key to env') 

# Initialize the language model (LLM) using the Groq API with Llama3.1
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=Config.LANGUAGE_MODEL_NAME)
prompt = ChatPromptTemplate.from_template(Prompts.MAIN_PROMPT.value)




class Chain: 
    def __init__(self):
       self.llm=llm
       self.prompt=prompt
    #    self.input_prompt=input_prompt


    def process_chain(self,input_prompt):
    # Create a chain to retrieve and process relevant documents using the LLM and prompt
        document_chain = create_stuff_documents_chain(self.llm, self.prompt)

        # Retrieve the vector store created from the PDF documents
        retriever = st.session_state.vectors.as_retriever()
        # info=st.info("")
        # Clear the info box once the processing starts
        # info.write("")

        # Create a retrieval chain that combines the document retriever and the language model
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Measure the time taken to generate the response
        start = time.process_time()
        response = retrieval_chain.invoke({"input": input_prompt})
        print("Response time:", time.process_time() - start)

        # Display the response (answer to the user's question)
        st.write(response["answer"])
        # info.write(" ")
        # Use an expander to show relevant document chunks for similarity search
        with st.expander("Document Similarity Search"):
            # Display relevant document chunks that were used to generate the response
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")

