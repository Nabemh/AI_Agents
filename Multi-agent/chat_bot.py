import os
import gradio as gr
from dotenv import load_dotenv
from datasets import load_dataset
import pandas as pd
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

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

search_tool = TavilySearchResults(api_key=tavily_api_key, max_results=3)

research_tool = Tool(
    name="Research tool",
    func=lambda query: search_tool.run(f"Search online for {query}"),
    description="Get generic information about this and return in Structured bullet points."
                """- Under 'How you sign in to Google,' tap Password. You might need to sign in.
                   - Enter your new password, then tap Change Password.
                        - Tip: When you enter your password on mobile, the first letter isn't case sensitive."""
)

all_tools = support_tools + [research_tool]

support_agent = initialize_agent(
    tools=all_tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=ConversationBufferMemory(memory_key="chat_history"),
    verbose=True
)

def chat(message, chat_history):
    try:
        response = support_agent.run(message)
        chat_history.append((message, response))
        return response
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        chat_history.append((message, error_msg))
        return error_msg


gr.ChatInterface(chat).launch()