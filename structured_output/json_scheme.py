from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Literal, Optional
from pydantic import BaseModel, Field
import json

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
)
#load json file 
with open("json_schema.json", "r") as f:
    json_schema = json.load(f)


structured_model = model.with_structured_output(json_schema)


result = structured_model.invoke("""
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—the $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor
Stunning 200MP camera
Long battery life with fast charging
S-Pen support

Review by Sachin Nepal
""")

print(result["summary"])
print(result["pros"])
print(result["cons"])
print(result["name"])
print(result["sentiment"])
