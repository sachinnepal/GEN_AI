import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()


# -----------------------------
# 1. Load webpage
# -----------------------------

url = "https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421"

response = requests.get(
    url,
    headers={
        "User-Agent": "Mozilla/5.0"
    },
    timeout=20
)

response.raise_for_status()


# -----------------------------
# 2. Parse HTML
# -----------------------------

soup = BeautifulSoup(response.text, "html.parser")

# Remove unnecessary HTML
for tag in soup(["script", "style", "noscript"]):
    tag.decompose()

text = soup.get_text(
    separator="\n",
    strip=True
)


# -----------------------------
# 3. Convert to LangChain Document
# -----------------------------

doc = Document(
    page_content=text,
    metadata={
        "source": url
    }
)

print("Loaded characters:", len(doc.page_content))


# -----------------------------
# 4. Gemini model
# -----------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0
)


# -----------------------------
# 5. Prompt
# -----------------------------

prompt = PromptTemplate.from_template(
    """Answer the following question based only on the provided text.

Question:
{question}

Text:
{text}
"""
)


# -----------------------------
# 6. Output parser
# -----------------------------

parser = StrOutputParser()


# -----------------------------
# 7. LCEL chain
# -----------------------------

chain = prompt | model | parser


# -----------------------------
# 8. Ask question
# -----------------------------

response = chain.invoke({
    "question": "What is the product that we are talking about?",
    "text": doc.page_content
})


print("\nAnswer:")
print(response)