import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader, WebBaseLoader, CSVLoader

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini",
    temperature=0.7
)

pdf_loader = PyPDFLoader("policy.pdf")
web_loader = WebBaseLoader("")
csv_loader = CSVLoader("")

