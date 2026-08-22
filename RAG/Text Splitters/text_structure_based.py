from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
I would not spend time polishing the current placeholder screen.

Instead, I'd build a professional Nearby UI first, then connect it to location data in the next phase.

That way, when we add GPS support, we only need to replace the data source—not redesign the screen.

"""


splitter =RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap =0,
)
result =splitter.split_text(text)
print(result)