from langchain_core.documents import Document
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

prompt= PromptTemplate(
    template='write a summery  on following poem\n {poem}',
    input_varaibles=['poem']
)

parser =StrOutputParser()

with open("cricket.txt", "r", encoding="utf-8") as f:
    text = f.read()

docs = [
    Document(
        page_content=text,
        metadata={"source": "cricket.txt"}
    )
]
chain = prompt | model | parser
result =chain.invoke({'poem':docs[0].page_content})
print(result)
# print(docs[0].page_content)
# print(docs[0].metadata)