from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# chat_template = ChatPromptTemplate.from_messages([
#     SystemMessage(content="You are a Helpful {domain} Expert"),
#     HumanMessage(content="Explain in simple terms, what is {topic}")
# ])
            #correct approach
chat_template = ChatPromptTemplate([
    ("system", "You are a Helpful {domain} Expert"),
    ("human", "Explain in simple terms, what is {topic}")
])

prompt = chat_template.invoke({
    "domain": "cricket",
    "topic": "LBW"
})

print(prompt)
