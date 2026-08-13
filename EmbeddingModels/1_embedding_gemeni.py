from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    dimensions=32
)

text1 = "Kathmandu is the capital of Nepal"
text2 = "Nepal's capital city is Kathmandu"

v1 = embedding.embed_query(text1)
v2 = embedding.embed_query(text2)

print(len(v1))
print(len(v2))