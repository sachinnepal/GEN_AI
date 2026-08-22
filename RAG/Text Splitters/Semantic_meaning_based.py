import os
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv


load_dotenv()
# Gemini embedding model
embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    dimensions=32
)

text = """
Python is a popular programming language used for web development,
data science, artificial intelligence, and automation.

Python uses simple and readable syntax. This makes it easy for
beginners to learn and for experienced developers to maintain.

Machine learning is a branch of artificial intelligence.
It allows computers to learn patterns from data without being
explicitly programmed for every task.

Neural networks are commonly used in modern machine learning.
They are inspired by the way biological neurons process information.

The weather in Kathmandu is generally warm during the summer.
The monsoon season brings significant rainfall to Nepal.
"""

# Create semantic splitter
splitter = SemanticChunker(
    embedding,
    breakpoint_threshold_type="percentile"
)

# Split based on semantic meaning
chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)