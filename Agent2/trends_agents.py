import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, initialize_agent

import tweepy

load_dotenv()

bearer_token = os.getenv("BEARER")
client = tweepy.Client(bearer_token=bearer_token)

api_key = os.getenv("GOOGLE_API_KEY")


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0.7,
    max_tokens=500,
    )

def get_user_tweets(username, max_results=10):
    user=client.get_user(username=username)
    user_id = user.data.id