from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    dimensions=32
)
documents = [
    "Kathmandu is the capital of Nepal",
    "Nepal's capital city is Kathmandu"
]

result = embedding.embed_documents(documents)

print(result)
print("Number of documents:", len(result))
print("Dimensions of first vector:", len(result[0]))