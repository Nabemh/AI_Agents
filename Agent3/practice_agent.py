import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0.2
    )

prompts = {
    "child": PromptTemplate.from_template("Explain {concept} like I'm 5. Use 1 sentence."),
    "teen": PromptTemplate.from_template("Summarize {concept} for a high school student in 2 sentences."),
    "expert": PromptTemplate.from_template("Describe {concept} in technical terms for a PhD. Use 3 bullet points.")
}


chains = {
    level: LLMChain(llm=llm, prompt=prompt)
    for level, prompt in prompts.items()
}

def explain(concept, level):
    return chains[level].run(concept=concept)

print(explain("chains in langchain", "teen"))