from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel
)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0
)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)


def count_text(text):
    return len(text.split())


joke_gen_chain = prompt | model | parser

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "wordcount": RunnableLambda(count_text)
})

final_chain = joke_gen_chain | parallel_chain

result = final_chain.invoke({
    "topic": "College"
})

print(result)