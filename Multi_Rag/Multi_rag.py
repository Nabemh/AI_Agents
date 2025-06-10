import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PyPDFLoader, WebBaseLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini",
    temperature=0.7
)

pdf_loader = PyPDFLoader("policy.pdf")
web_loader = WebBaseLoader("https://www.typetone.ai/blog/faq-examples-100-faqs-for-your-landing-page")
csv_loader = CSVLoader("LIST OF Physican and Virtual attendance.xlsx - Physical Attandance.csv")

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

embeddings = ChatGoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("multidoc_index")

def retrieve ():
    return

prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY these sources. Cite each source like [1], [2].

Sources:
{context}

Question: {question}
""")

rag_chain = prompt | llm

def ask(query, sources=None):
    docs = retrieve(query, sources)
    context = "\n\n".join(
        f"[{i+1}] {d.page_content} (Source: {d.metadata['source']})"
        for i, d in enumerate(docs)
    )
    return rag_chain.invoke({"context": context, "question": query})