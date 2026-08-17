from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


load_dotenv()


# Models
model1 = GoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0
)

model2 = GoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0
)


# Prompt 1 - Generate Notes
prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text:\n{text}",
    input_variables=["text"]
)


# Prompt 2 - Generate Quiz
prompt2 = PromptTemplate(
    template="Generate 5 short questions from the following text:\n{text}",
    input_variables=["text"]
)


# Prompt 3 - Merge Notes and Quiz
prompt3 = PromptTemplate(
    template="""Merge the provided notes and quiz into a single document.

Notes:
{notes}

Quiz:
{quiz}
""",
    input_variables=["notes", "quiz"]
)


# Output parser
parser = StrOutputParser()


# Run both chains in parallel
parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser
})


# Final chain
chain3 = prompt3 | model1 | parser


# Complete chain
chain4 = parallel_chain | chain3


# Input text
text = """
My project is a Nepali Voice AI Agent that allows users to communicate
with an AI through Nepali speech or text. When the user speaks, Google
Speech-to-Text converts the Nepali speech into text. Gemini then understands
the user's intent and generates a response. LangGraph controls the agent
workflow and decides whether the system needs to retrieve information through
RAG or call an external API. PostgreSQL and pgvector provide persistent and
vector-based storage, while Redis manages temporary conversation state.
Finally, Google Cloud Text-to-Speech converts the generated Nepali response
into audio, which is played back to the user. The research will evaluate the
system using STT accuracy, response quality, task completion, latency, TTS
quality and user satisfaction, while also analyzing the operational cost
of the system.
"""


# Invoke complete chain
result = chain4.invoke({
    "text": text
})


print(result)