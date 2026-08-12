from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model=ChatAnthropic(
    model="claude_sonet",
    temperature=0
)
result=model.invoke("Who is Sachin Nepal")
print(result.text)

