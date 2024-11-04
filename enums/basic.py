from enum import Enum 

class StreamlitConfig(Enum):
    PAGE_ICON_PATH = "/workspaces/Rag_ht/app_data/icon.png"
    FALLBACK_ICON_PATH = "app_data/icon.png"
    PAGE_TITLE = "Heat Treatment Chatbot"
    LAYOUT = "wide"
    SIDEBAR_IMAGE_URL = "https://www.win-therm.com.my/wp-content/uploads/2021/02/1-1251ok-before-1024x774.jpg"
    HEADER="Chat with Heat Treatment PDFs"
    SIDEBAR_MARKDOWN="""
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
    
class Prompts(Enum):
    MAIN_PROMPT = """
    Answer the questions based on the provided context. If the provided context does not include the answer, \
    provide an answer based on your knowledge.
    {context}
    <context>
    Questions: {input}
    """