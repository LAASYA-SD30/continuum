import requests

API_BASE = "http://127.0.0.1:8000"
API_KEY = "test-key-123"
HEADERS = {"x-api-key": API_KEY}

def book_slot(day, time):
    fact_text = f"User booked a meeting slot on {day} at {time}"
    payload = {
        "fact": fact_text,
        "source": "scheduler_tool",
        "confidence": 1.0,
        "pinned": False,
        "expires_at": None
    }
    response = requests.post(f"{API_BASE}/write", json=payload, headers=HEADERS)
    print(f"Scheduler saved: {fact_text}")
    print(f"Response: {response.json()}")

if __name__ == "__main__":
    print("Scheduler Tool Demo (not a chatbot — simulates a booking action)\n")
    day = input("Enter a day (e.g. Monday): ")
    time = input("Enter a time (e.g. 3 PM): ")
    book_slot(day, time)