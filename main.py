from fastapi import FastAPI
from pydantic import BaseModel
from supabase import create_client
from typing import Optional
from datetime import datetime, timezone
import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()
app = FastAPI()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
embedder = SentenceTransformer('all-MiniLM-L6-v2')

class Fact(BaseModel):
    developer_id: str
    fact: str
    source: str
    confidence: float
    pinned: Optional[bool] = False
    expires_at: Optional[datetime] = None

@app.post("/write")
def write_fact(fact: Fact):
    existing = supabase.table("memories") \
        .select("*") \
        .eq("developer_id", fact.developer_id) \
        .execute()

    for row in existing.data:
        if fact.fact.lower() in row["fact"].lower() or row["fact"].lower() in fact.fact.lower():
            continue
        words_new = set(fact.fact.lower().split())
        words_old = set(row["fact"].lower().split())
        overlap = words_new.intersection(words_old)
        if len(overlap) >= 2 and row["fact"].lower() != fact.fact.lower():
            return {
                "status": "conflict",
                "message": f"You previously said: '{row['fact']}'. Update this to: '{fact.fact}'?",
                "existing_fact_id": row["id"],
                "new_fact": fact.fact
            }

    fact_data = json.loads(fact.json())
    fact_data["embedding"] = embedder.encode(fact.fact).tolist()

    result = supabase.table("memories").insert(fact_data).execute()
    return {"status": "saved", "data": result.data}

@app.get("/retrieve/{developer_id}")
def retrieve_facts(developer_id: str):
    now = datetime.now(timezone.utc).isoformat()
    result = supabase.table("memories") \
        .select("*") \
        .eq("developer_id", developer_id) \
        .or_(f"expires_at.is.null,expires_at.gt.{now},pinned.eq.true") \
        .execute()
    return result.data

@app.get("/search/{developer_id}")
def search_facts(developer_id: str, query: str):
    now = datetime.now(timezone.utc).isoformat()
    all_facts = supabase.table("memories") \
        .select("*") \
        .eq("developer_id", developer_id) \
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
    return results[:5]

class ConflictResolution(BaseModel):
    existing_fact_id: str
    new_fact: str
    developer_id: str
    source: str
    confidence: float

@app.post("/resolve-conflict")
def resolve_conflict(resolution: ConflictResolution):
    supabase.table("memories").delete().eq("id", resolution.existing_fact_id).execute()

    new_fact = {
        "developer_id": resolution.developer_id,
        "fact": resolution.new_fact,
        "source": resolution.source,
        "confidence": resolution.confidence,
        "pinned": False,
        "expires_at": None,
        "embedding": embedder.encode(resolution.new_fact).tolist()
    }
    result = supabase.table("memories").insert(new_fact).execute()
    return {"status": "updated", "data": result.data}

@app.delete("/memory/{fact_id}")
def delete_fact(fact_id: str):
    result = supabase.table("memories").delete().eq("id", fact_id).execute()
    return {"status": "deleted", "data": result.data}

class FactUpdate(BaseModel):
    fact: Optional[str] = None
    confidence: Optional[float] = None
    pinned: Optional[bool] = None
    expires_at: Optional[datetime] = None

@app.patch("/memory/{fact_id}")
def update_fact(fact_id: str, update: FactUpdate):
    update_data = json.loads(update.json(exclude_none=True))
    if update.fact is not None:
        update_data["embedding"] = embedder.encode(update.fact).tolist()
    result = supabase.table("memories").update(update_data).eq("id", fact_id).execute()
    return {"status": "updated", "data": result.data}
@app.get("/export/{developer_id}")
def export_memories(developer_id: str):
    result = supabase.table("memories") \
        .select("id, fact, source, confidence, pinned, expires_at, created_at") \
        .eq("developer_id", developer_id) \
        .execute()

    export_data = {
        "developer_id": developer_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "format_version": "1.0",
        "total_facts": len(result.data),
        "memories": result.data
    }
    return export_data