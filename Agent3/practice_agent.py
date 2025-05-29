import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0.7
    )

prompt = ChatPromptTemplate.from_template(
    "Explain {concept} in a way a five year old can understand. Use less than 100 words"
)

final_prompt = prompt.format(concept="Subnetting")

chain = LLMChain(llm=llm, prompt=prompt)

result = chain.run(concept = "Subnetting")
print(result)