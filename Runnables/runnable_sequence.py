from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0
   
)

prompt = PromptTemplate(
    template="Write a Joke on this {topic}",
    input_variables=['topic']
)
parser= StrOutputParser()
#runnable sequence
chain =RunnableSequence( prompt,model,parser)

result = chain.invoke({'topic':'cricket'})
print (result)

