from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    max_output_tokens = 100
)

result = llm.invoke("Write a 5 line poem?")
# print(result.content)

with open("output.txt", "w", encoding="utf-8") as file:
    file.write(result.text)


    

