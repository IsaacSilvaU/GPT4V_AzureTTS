import os
from openai import OpenAI
client = OpenAI()

OPENAI_API_KEY = ""
client.api_key = os.getenv("OPENAI_API_KEY")

completion = client.chat.completions.create(
  model="gpt-3.5-turbo-1106",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)

print(completion.choices[0].message)