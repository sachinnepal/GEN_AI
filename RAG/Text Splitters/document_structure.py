from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

text = """
def get_age():
    while True:
        age = int(input("Enter your age: "))

        if 0 <= age <= 120:
            return age

        print("Invalid age. Try again.")


def check_adult(age):
    if age >= 18:
        return "You are an adult."
    return "You are a minor."


name = input("Enter your name: ")
age = get_age()

message = check_adult(age)

print(f"\nHello {name}")
print(message)
"""


splitter =RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200,
    chunk_overlap =0,
)
result =splitter.split_text(text)
print(result)