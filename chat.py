from groq import Groq
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

claude = Groq(api_key=os.getenv("GROQ_API_KEY"))

API_BASE = "http://127.0.0.1:8000"
API_KEY = "test-key-123"
HEADERS = {"x-api-key": API_KEY}

def save_memory(fact_text):
    payload = {
        "fact": fact_text,
        "source": "chat_demo",
        "confidence": 0.8,
        "pinned": False,
        "expires_at": None
    }
    requests.post(f"{API_BASE}/write", json=payload, headers=HEADERS)

def search_memories(query):
    response = requests.get(f"{API_BASE}/search", params={"query": query}, headers=HEADERS)
    return response.json()

def extract_and_save_facts(user_message, retry=True):
    prompt = f"""Read this message and extract any personal facts, preferences, or details worth remembering about the user. Return ONLY a JSON list of short fact strings, nothing else — no explanation, no markdown formatting. If there is nothing worth remembering, return an empty list [].

Message: "{user_message}"

Example output: ["User prefers tea over coffee", "User is a final-year AI/ML student"]"""

    response = claude.chat.completions.create(
        model="openai/gpt-oss-20b",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )

    raw_output = response.choices[0].message.content.strip()

    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`")
        if raw_output.startswith("json"):
            raw_output = raw_output[4:]
        raw_output = raw_output.strip()

    try:
        facts = json.loads(raw_output)
        for fact in facts:
            save_memory(fact)
        return facts
    except Exception as e:
        if retry:
            return extract_and_save_facts(user_message, retry=False)
        print(f"[DEBUG] Extraction failed after retry: {e}\n")
        return []

def chat():
    print("Continuum Chat Demo — type 'quit' to exit\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        memories = search_memories(user_input)
        memory_summary = [(m['fact'], round(m['similarity'], 3)) for m in memories]
        print(f"\n[DEBUG] Search results: {memory_summary}\n")
        memory_context = "\n".join([f"- {m['fact']}" for m in memories])

        system_prompt = f"You are a helpful assistant. Here is what you remember about this user:\n{memory_context}\n\nUse this context naturally if relevant."

        response = claude.chat.completions.create(
            model="openai/gpt-oss-20b",
            max_tokens=500,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )

        print(f"\nClaude: {response.choices[0].message.content}\n")

        new_facts = extract_and_save_facts(user_input)
        if new_facts:
            print(f"[Memory saved: {new_facts}]\n")

if __name__ == "__main__":
    chat()