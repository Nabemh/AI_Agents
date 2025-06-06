import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from datasets import load_dataset

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

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

def semantic_faq_search(query):
    #look for most similar FAQ
    docs = loaded_store.similarity_search(query, k=1)
    if docs:
        matched_question = docs[0].page_content
        #Get corresponding answer from DataFrame
        return df[df["question"] == matched_question]["answer"].values[0]
    return "No relevant FAQ found"

print(semantic_faq_search("How do I change my password?"))