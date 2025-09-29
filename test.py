import os
from dotenv import load_dotenv
from openai import OpenAI

# Load API key
load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Sample prompt
prompt = "Hello! Please respond with a short, fun greeting."

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    print("✅ API key works, prompt response:")
    print(response.choices[0].message.content)

except Exception as e:
    print("❌ API key test failed:", e)
