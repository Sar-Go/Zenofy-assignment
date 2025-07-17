from openai import OpenAI
import os

OPENAI_API_KEY = "sk-proj-HzHFLXON8v3z9n71gZZN6hRkdDwVlz0QBSNmYY9vmYqmNNxgPT3BlbkFJqsZhCOi0Yvs9kuqckavSVwO0oAKNI_Y4WgbZYGkijryR1XQPk6bGyI8yFnmbNHcy7PhtTcYIQA"
client = OpenAI(api_key=OPENAI_API_KEY)
from llama_index.core import SimpleDirectoryReader, GPTVectorStoreIndex

documents = ["./file.txt"]

reader = SimpleDirectoryReader(".")
index = GPTVectorStoreIndex.from_documents(reader.load_data())

query_engine = index.as_query_engine(similarity_top_k=3)

def generate_response(query):
    retrieved_info = query_engine.query(query)
    prompt = f"Using the following information: {retrieved_info}, please answer the question: {query}"
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Example query
query = "My colleague- Alice made a suspicious trade. What should be done?"
response = query_engine.query(query)
print(response)

query2 = "How many days i have to disclose trades made by Mr Avinash who is the director?"
print(generate_response(query2))