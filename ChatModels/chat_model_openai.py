from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-5.1",
    temperature=0
)

result = model.invoke("Who is Sachin Nepal?")

print(result.content)