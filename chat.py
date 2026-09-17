from groq import Groq
from supabase import create_client
import os
import json
from dotenv import load_dotenv
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
claude = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedder = SentenceTransformer('all-MiniLM-L6-v2')

DEVELOPER_ID = "chat_demo_user"

def save_memory(fact_text):
    new_fact = {
        "developer_id": DEVELOPER_ID,
        "fact": fact_text,
        "source": "chat_demo",
        "confidence": 0.8,
        "pinned": False,
        "expires_at": None,
        "embedding": embedder.encode(fact_text).tolist()
    }
    supabase.table("memories").insert(new_fact).execute()

def search_memories(query):
    now = datetime.now(timezone.utc).isoformat()
    all_facts = supabase.table("memories") \
        .select("*") \
        .eq("developer_id", DEVELOPER_ID) \
        .or_(f"expires_at.is.null,expires_at.gt.{now},pinned.eq.true") \
        .execute()

    query_embedding = embedder.encode(query)

    results = []
    for row in all_facts.data:
        if row.get("embedding"):
            raw_embedding = row["embedding"]
            if isinstance(raw_embedding, str):
                raw_embedding = json.loads(raw_embedding)
            fact_embedding = np.array(raw_embedding, dtype=float)
            similarity = np.dot(query_embedding, fact_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(fact_embedding)
            )
            results.append({**row, "similarity": float(similarity)})

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:3]

def extract_and_save_facts(user_message):
    prompt = f"""Read this message and extract any personal facts, preferences, or details worth remembering about the user. Return ONLY a JSON list of short fact strings, nothing else. If there is nothing worth remembering, return an empty list [].

Message: "{user_message}"

Example output: ["User prefers tea over coffee", "User is a final-year AI/ML student"]"""

    response = claude.chat.completions.create(
        model="openai/gpt-oss-20b",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )

    raw_output = response.choices[0].message.content
    try:
        facts = json.loads(raw_output)
        for fact in facts:
            save_memory(fact)
        return facts
    except Exception as e:
        print(f"[DEBUG] Extraction failed: {e}\n")
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