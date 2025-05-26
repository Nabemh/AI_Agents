import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, initialize_agent

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.7,
    google_api_key=api_key
    )

def echo_tool(input : str) -> str:
    return input

tools = [
    Tool(
        name="Echo",
        func=echo_tool,
        description="Echoes back the input that has been passed"
    )
]

agent = initialize_agent(
    tools, 
    llm, 
    agent="zero-shot-react-description", 
    verbose=True
)

response = agent.run("Hello, what can you do?")
print(response)