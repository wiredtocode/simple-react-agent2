
import os

from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "How to be an ai engineer",
        }
    ],
    model="openai/gpt-oss-20b",
)

print(chat_completion.choices[0].message.content)