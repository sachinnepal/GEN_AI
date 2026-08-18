from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# Model
model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0
)

# Parser
parser = StrOutputParser()

# Prompt
summary_prompt = PromptTemplate(
    template="Give me a 3-line summary about {topic}",
    input_variables=["topic"]
)

# Summary chain
summary_chain = summary_prompt | model | parser

# Parallel chain
parallel_chain = RunnableParallel(
    original=RunnablePassthrough(),
    summary=summary_chain
)

# Invoke
result = parallel_chain.invoke({
    "topic": "Artificial Intelligence"
})

# Output
print("Original Input:")
print(result["original"])

print("\nSummary:")
print(result["summary"])