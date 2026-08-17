
#NOT WORKING NOW.

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers.structured import StructuredOutputParser , ResponseSchema

load_dotenv()

# Initialize the model with enough tokens for the JSON response
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_output_tokens=500
)

# Define schemas with unique keys
schema = [
    ResponseSchema(name='fact_1', description='fact-1 about the topic'),
    ResponseSchema(name='fact_2', description='fact-2 about the topic'),
    ResponseSchema(name='fact_3', description='fact-3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

# Correct string syntax and method call (get_format_instructions)
template = PromptTemplate(
    template="Give 3 facts about {topic}.\n{format_instructions}",
    input_variables=['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

# Modern LangChain Expression Language (LCEL) chain
chain = template | model | parser

# Invoke chain and receive parsed dict directly
final_result = chain.invoke({'topic': 'Black Hole'})

# Print keys or the full dictionary directly
print(final_result)
print("Fact 1:", final_result['fact_1'])