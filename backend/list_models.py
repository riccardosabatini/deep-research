import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def list_models():
    provider = os.getenv("AI_PROVIDER", "openai").lower()
    api_key = os.getenv("AI_API_KEY")
    base_url = os.getenv("AI_BASE_URL")

    print(f"--- Checking Models for Provider: {provider} ---")
    if base_url:
        print(f"Base URL: {base_url}")

    try:
        if provider in ["openai", "deepseek", "openrouter", "groq", "ollama", "together", "xai"]:
            # Use OpenAI Client for compatible providers
            from openai import OpenAI
            
            # Adjust base_url for specific providers if not set
            if not base_url:
                if provider == "deepseek":
                    base_url = "https://api.deepseek.com"
                elif provider == "openrouter":
                    base_url = "https://openrouter.ai/api/v1"
                elif provider == "groq":
                    base_url = "https://api.groq.com/openai/v1"
                elif provider == "ollama":
                    base_url = "http://localhost:11434/v1"
            
            client = OpenAI(api_key=api_key, base_url=base_url)
            models = client.models.list()
            
            print(f"\nFound {len(models.data)} models:")
            for model in sorted(models.data, key=lambda x: x.id):
                print(f" - {model.id}")

        elif provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            # Anthropic doesn't have a public list models endpoint in the same way, 
            # but we can try to list recent ones or just print a message.
            # Actually, newer SDK versions might support it, but it's less standard.
            # Let's check if we can list. 
            # As of late 2024, Anthropic added a models endpoint.
            try:
                # This might fail on older SDKs
                models = client.models.list()
                print(f"\nFound {len(models.data)} models:")
                for model in sorted(models.data, key=lambda x: x.id):
                    print(f" - {model.id}")
            except Exception:
                print("\nCould not list models automatically for Anthropic (endpoint might not be available in this SDK version).")
                print("Common models: claude-3-5-sonnet-latest, claude-3-5-haiku-latest, claude-3-opus-latest")

        elif provider in ["google", "google_genai"]:
            import google.generativeai as genai
            if api_key:
                genai.configure(api_key=api_key)
            
            print("\nAvailable Google Models:")
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    print(f" - {m.name}")

        else:
            print(f"Unknown or unsupported provider for listing models: {provider}")
            print("Supported for listing: openai, deepseek, openrouter, groq, ollama, anthropic, google")

    except Exception as e:
        print(f"\nError fetching models: {e}")
        print("Please check your API key and configuration in .env")

if __name__ == "__main__":
    list_models()
