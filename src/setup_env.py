import os
from dotenv import set_key, load_dotenv

def setup_groq_api_key():
    load_dotenv()
    existing = os.getenv("GROQ_API_KEY")
    if existing:
        print(f"Existing Groq API key found: {existing[:6]}...{existing[-4:]}")
        if input("Keep existing key? (y/n): ").lower() == "y":
            return
    key = input("Enter your Groq API key: ").strip()
    if key:
        set_key(".env", "GROQ_API_KEY", key)
        print("Saved successfully.")
