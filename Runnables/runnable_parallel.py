from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0
)

parser = StrOutputParser()

summary_prompt = PromptTemplate(
    template="Give me a 3-line summary about {topic}",
    input_variables=["topic"]
)

keypoints_prompt = PromptTemplate(
    template="Give me 5 key points about {topic}",
    input_variables=["topic"]
)

facts_prompt = PromptTemplate(
    template="Give me 3 interesting facts about {topic}",
    input_variables=["topic"]
)

summary_chain= summary_prompt | model | parser
keypoints_chain= keypoints_prompt | model | parser
facts_chain= facts_prompt | model | parser

parallel_chain=RunnableParallel(
    summary=summary_chain,
    keypoints=keypoints_chain,
    facts=facts_chain
)
 
result= parallel_chain.invoke({'Football'})
print("Summary:")
print(result["summary"])

print("\nKey Points:")
print(result["keypoints"])

print("\nFacts:")
print(result["facts"])