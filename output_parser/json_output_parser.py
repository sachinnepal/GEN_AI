from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import  PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
parser = JsonOutputParser()

model=ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    max_output_tokens = 100
)

template= PromptTemplate(
    template='Give me 5 fact about {topic}/n {format_instruction}',
    input_variables= ['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}

)
chain = template | model | parser
# prompt=template.format()
# print(prompt)

result=chain.invoke({'topic':'blackhole'})

print(result)