from ollama import Client
messages = [{"role": "system", "content": "You are a joker. reply to everything sarcastically"},
            {"role": "user", "content": "Where is my cake?"},
            {"role": "user", "content": "Who are you?"}
            ]

client = Client()

responses = client.chat(
    model='smollm2',
    messages=messages
)

print(responses.messages)
