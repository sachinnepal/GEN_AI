from langchain_huggingface import HuggingFaceEmbeddings

embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

#documents

documents = [
    "Kathmandu is the capital of Nepal",
    "Nepal's capital city is Kathmandu"
]

result = embedding.embed_documents(documents)

print(result)
print("Number of documents:", len(result))
print("Dimensions of first vector:", len(result[0]))
