from pypdf import PdfReader
from langchain_core.documents import Document

reader = PdfReader("cricket.pdf")

docs = []

for page_number, page in enumerate(reader.pages):
    text = page.extract_text()

    docs.append(
        Document(
            page_content=text or "",
            metadata={
                "source": " cricket.pdf",
                "page": page_number
            }
        )
    )

print(docs)