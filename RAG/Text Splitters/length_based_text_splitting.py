from langchain_text_splitters import CharacterTextSplitter

text = """
I would not spend time polishing the current placeholder screen.

Instead, I'd build a professional Nearby UI first, then connect it to location data in the next phase.

That way, when we add GPS support, we only need to replace the data source—not redesign the screen.

"""


splitter =CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap =5,
    separator=''
)
result =splitter.split_text(text)
print(result)





