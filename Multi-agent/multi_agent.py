import os
from dotenv import load_dotenv
from datasets import load_dataset
import pandas as pd
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0.7,
)

dataset = load_dataset("MakTek/Customer_support_faqs_dataset", split="train")
df = dataset.to_pandas()


def search_knowledge_base(query):
    results = df[df["question"].str.contains(query, case=False, na=False)]
    if not results.empty:
        return results.iloc[0]["answer"]
    else:
        return "Sorry, I couldn't find an answer to that. Would you like for this to be escalated?"

support_tools = [
    Tool(
        name="FAQ Search",
        func=search_knowledge_base,
        description="Answers FAQs about products, policies, etc., You are a helpful support assistant. Use the FAQ to answer user questions"
    )
]

support_agent = initialize_agent(
    tools=support_tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=ConversationBufferMemory(memory_key="chat_history"),
    verbose=True
)

response = support_agent.run("How do I reset my password?")
print(response)

