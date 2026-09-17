from supabase import create_client
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
embedder = SentenceTransformer('all-MiniLM-L6-v2')

DEVELOPER_ID = "chat_demo_user"

def book_slot(day, time):
    fact_text = f"User booked a meeting slot on {day} at {time}"
    new_fact = {
        "developer_id": DEVELOPER_ID,
        "fact": fact_text,
        "source": "scheduler_tool",
        "confidence": 1.0,
        "pinned": False,
        "expires_at": None,
        "embedding": embedder.encode(fact_text).tolist()
    }
    supabase.table("memories").insert(new_fact).execute()
    print(f"Scheduler saved: {fact_text}")

if __name__ == "__main__":
    print("Scheduler Tool Demo (not a chatbot — simulates a booking action)\n")
    day = input("Enter a day (e.g. Monday): ")
    time = input("Enter a time (e.g. 3 PM): ")
    book_slot(day, time)