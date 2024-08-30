from openai import OpenAI

client = OpenAI()

completion = client.chat.completions.create(
model="gpt-3.5-turbo",
messages=[{"role": "system", "content": "You are a helpful assistant."},
{"role": "user", "content": "Hello!"}
])

print(completion.choices[0].message)


response = client.chat.completions.create (
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Who was the first man on the moon?"},
        {"role": "assistant", "content": "The first man on the moon was Neil Armstrong."},
        {"role": "user", "content": "Tell me more about him."}
    ],
    top_p=0.5,temperature=0.8,n=3,stop="Answer:",frequency_penalty=0.6,presence_penalty=0.8
)

 
