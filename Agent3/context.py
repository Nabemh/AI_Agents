import os
import gradio as gr
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
     model="gemini-2.0-flash-001",
     temperature=0.7,
     max_tokens=200
)


memory = ConversationBufferMemory(memory_key="chat_history", input_key="concept")


prompt = ChatPromptTemplate.from_template(
    """
    Chat history: \n{chat_history}

    \n\nUser: {concept}

    \nAI:

    """
)


chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

def chat(message):
    response = chain.run(message)
    return response

gr.ChatInterface(chat).launch()