from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=1.5,
)

class Person(BaseModel):
    name:str=Field(description='Name of the person')
    age:int=Field(gt=18,description='age of the person')
    city:str=Field(description='write the name of the city of that person')

parser=PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="""Generate the name, age, and city of a fictional {place} person.

{format_instruction}""",
    input_variables=["place"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)
# prompt = template.invoke({'place':'Nepali'})
# print(prompt)
# result = model.invoke(prompt)
# final_result = parser.parse(result.text)
# print(final_result)

#using chains

chain = template | model | parser
final_result=chain.invoke({'place':'Nepali'})
print(final_result)
