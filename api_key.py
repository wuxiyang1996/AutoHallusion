
try:
    with open("your path", "r") as f:
        OPENAI_API_KEY = f.read()
except:
    OPENAI_API_KEY = 'your_openai_api_key'