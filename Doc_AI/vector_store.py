import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate


load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0.7,
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

loader = PyPDFLoader("Doc_AI/NIST.CSWP.29.pdf")
pages = loader.load_and_split()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
chunks = splitter.split_documents(pages)

pdf_store = FAISS.from_documents(chunks, embeddings)

def pdf_qa(query):
    docs = pdf_store.similarity_search(query)
    context = "\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on this context:
    {context}

    Question: {question}
    """)
    chain = prompt | llm
    return chain.invoke ({"context": context, "question": query})

if __name__ == "__main__":
    while True:
        query = input("Ask a question about the PDF (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        answer = pdf_qa(query)
        print("\nAnswer:", answer, "\n")
