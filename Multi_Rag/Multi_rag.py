import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader, WebBaseLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini",
    temperature=0.7
)

pdf_loader = PyPDFLoader("policy.pdf")
web_loader = WebBaseLoader("")
csv_loader = CSVLoader("")

pdf_pages = pdf_loader.load()
web_pages = web_loader.load()
csv_doc = csv_loader.load()

all_docs = pdf_pages + web_pages + csv_doc



splitter = RecursiveCharacterTextSplitter( 
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " "]
)

chunks = splitter.split_documents(all_docs)