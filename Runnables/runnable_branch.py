from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel,
    RunnableBranch
)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0
)

prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarize the following text:\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain1 = prompt1 | model | parser

chain2 = RunnableBranch(
    (
        lambda x: len(x.split()) > 1000,
        prompt2 | model | parser
    ),
    RunnablePassthrough()
)

finalchain = chain1 | chain2

result = finalchain.invoke({
    "topic": "Russia vs Ukraine"
})

print(result)