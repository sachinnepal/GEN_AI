from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import  PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

model=ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    max_output_tokens = 100
)

template1= PromptTemplate(
    template='write a detail report on {topic}',
    input_varaibles=['topic']
)
template2= PromptTemplate(
    template='write a 5 line summery on following text./n {text}',
    input_varaibles=['text']
)


parser= StrOutputParser()

chain =template1 | model | parser | template2| model |parser

result =chain.invoke({'topic':'Black Hole'})

print (result)



    