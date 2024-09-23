
try:
    with open("your path", "r") as f:
        GEMINI_API_KEY = f.read()
except:
    GEMINI_API_KEY = 'your_openai_api_key'