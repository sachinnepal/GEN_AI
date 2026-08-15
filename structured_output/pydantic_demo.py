from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Literal, Optional
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
)


class Review(BaseModel):

    key_themes: list[str] = Field(
        description="Write down all the key themes discovered in the review"
    )

    summary: str = Field(
        description="A brief summary of the device review"
    )

    sentiment: Literal["pos", "neg", "neutral"] = Field(
        description="Return only pos, neg, or neutral"
    )

    pros: Optional[list[str]] = Field(
        description="Write down all the pros inside the list"
    )

    cons: Optional[list[str]] = Field(
        description="Write down all the cons inside the list"
    )


structured_model = model.with_structured_output(Review)


result = structured_model.invoke("""
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—the $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor
Stunning 200MP camera
Long battery life with fast charging
S-Pen support

Review by Nitish Singh
""")


print("SUMMARY:")
print(result.summary)

print("\nSENTIMENT:")
print(result.sentiment)

print("\nKEY THEMES:")
print(result.key_themes)

print("\nPROS:")
print(result.pros)

print("\nCONS:")
print(result.cons)