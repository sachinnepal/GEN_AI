from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    PydanticOutputParser,
    StrOutputParser
)
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel
)


load_dotenv()


model = GoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0
)


# -----------------------------
# Pydantic Model
# -----------------------------

class Review(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Give the sentiment according to the review"
    )


# -----------------------------
# Pydantic Parser
# -----------------------------

parser = PydanticOutputParser(
    pydantic_object=Review
)


# -----------------------------
# Sentiment Prompt
# -----------------------------

prompt1 = PromptTemplate(
    template="""Classify the sentiment into Positive or Negative
for the following feedback:

{review}

{format_instruction}
""",
    input_variables=["review", "format_instruction"]
)


chain1 = prompt1 | model | parser


# -----------------------------
# Response Prompts
# -----------------------------

prompt2 = PromptTemplate(
    template="""Write ONLY one short response to the customer's positive feedback.

Do not provide multiple options.
Do not provide explanations.
Do not provide tips.
Do not use headings.
Do not use bullet points.

Customer feedback:
{feedback}
""",
    input_variables=["feedback"]
)


prompt3 = PromptTemplate(
    template="""Write ONLY one short response to the customer's negative feedback.

Do not provide multiple options.
Do not provide explanations.
Do not provide tips.
Do not use headings.
Do not use bullet points.

Customer feedback:
{feedback}
""",
    input_variables=["feedback"]
)


# -----------------------------
# Output Parser for responses
# -----------------------------

str_parser = StrOutputParser()


# -----------------------------
# Branch
# -----------------------------

branch_chain = RunnableBranch(

    (
        lambda x: x["sentiment"].sentiment == "positive",
        prompt2 | model | str_parser
    ),

    (
        lambda x: x["sentiment"].sentiment == "negative",
        prompt3 | model | str_parser
    ),

    RunnableLambda(
        lambda x: "Could not find sentiment"
    )
)


# -----------------------------
# Preserve review + sentiment
# -----------------------------

final_chain = (
    RunnableParallel(
        feedback=lambda x: x["review"],
        sentiment=chain1
    )
    | branch_chain
)


# -----------------------------
# Invoke
# -----------------------------

result = final_chain.invoke({
    "review": "This is a terrible phone",
    "format_instruction": parser.get_format_instructions()
})


print(result)