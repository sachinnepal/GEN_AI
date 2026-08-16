from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import  PromptTemplate

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
prompt1=template1.invoke({'topic':'Blackhole'})
result=model.invoke(prompt1)

prompt2=template2.invoke({'text':result.text})
result1=model.invoke(prompt2)

print([result1.text])






    