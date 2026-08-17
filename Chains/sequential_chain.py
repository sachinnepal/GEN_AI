from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detail report on {topic}',
    input_variables=['topic']
 )


prompt2 = PromptTemplate(
    template='Generate a 5 pointer summery from a foloowing text\n {text}',
    input_variables=['text']
 )

model = GoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0
   
)


parser = StrOutputParser()
chain = prompt1 | model | prompt2 | model | parser
result = chain.invoke({'topic':'unemployment in nepal'})
print(result)