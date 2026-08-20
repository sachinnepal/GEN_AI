from pathlib import Path
from langchain_pymupdf4llm import PyMuPDF4LLMLoader

DOCUMENTS_DIR = Path(__file__).parent / "documents"

documents = []

for pdf_file in DOCUMENTS_DIR.rglob("*.pdf"):
    print(f"Loading: {pdf_file}")

    loader = PyMuPDF4LLMLoader(str(pdf_file))
    docs = loader.load()

    documents.extend(docs)

print("\n-----------------------------")
print(f"Total pages loaded: {len(documents)}")
print("-----------------------------")

for i, doc in enumerate(documents):
    print(f"\nDocument {i + 1}")
    print("Source:", doc.metadata.get("source"))
    print("Page:", doc.metadata.get("page"))
    print(doc.page_content[:300])