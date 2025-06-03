import os
import gradio as gr
from dotenv import load_dotenv
from datasets import load_dataset
import pandas as pd
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0.7,
)

#loading dataset
dataset = load_dataset("MakTek/Customer_support_faqs_dataset", split="train")
df = dataset.to_pandas()
texts = df["question"].tolist()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#prepare/create vectorstore
vectorstore = FAISS.from_texts(texts, embeddings)

#save/load for reuse
vectorstore.save_local("faiss_index")
loaded_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

loader = PyPDFLoader("policy.pdf")
pages = loader.load_and_split()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
chunks = splitter.split_documents(pages)

pdf_store = FAISS.from_documents(chunks, embeddings)

def pdf_qa(query):
    docs = pdf_store.similarity_search(query)
    context = "\n".join([d.page_content for d in docs])

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on this context:
    {context}

    Question: {question}
    """)
    chain = prompt | llm  # Pipe syntax (LangChain's LCEL)
    return chain.invoke({"context": context, "question": query})


def semantic_faq_search(query):
    #look for most similar FAQ
    docs = loaded_store.similarity_search(query, k=1)
    if docs:
        matched_question = docs[0].page_content
        #Get corresponding answer from DataFrame
        return df[df["question"] == matched_question]["answer"].values[0]
    return "No relevant FAQ found"

support_tool = [
    Tool(
        name="Support Tool",
        func=semantic_faq_search,
        description="Runs a semantic search through FAQ to look for answers to questions"
    )
]