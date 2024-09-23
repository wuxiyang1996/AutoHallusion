try:
    with open("your path", "r") as f:
        CLAUDE_API_KEY = f.read()
except:
    CLAUDE_API_KEY = 'your_claude_api_key'