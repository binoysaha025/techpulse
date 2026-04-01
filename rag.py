import os 
import requests
import anthropic
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

#clients
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("techpulse")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
claude = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))

TRIBE_URL = "http://127.0.0.1:8000/score"  # running FastAPI server

#1 retrieve top chunks from Pinecone
def retrieve(query: str, top_k=50) -> list[dict]:
    #embed query into vector space
    q_vec = embedder.encode(query).tolist()

    #cosine similarity search in Pinecone; returns top k most similar chunks
    results = index.query(vector=q_vec, top_k=top_k, include_metadata=True)

    chunks = []
    for match in results["matches"]:
        chunks.append({
            "text": match["metadata"]["text"],
            "title": match["metadata"]["title"],
            "url": match["metadata"]["url"],
            "outlet": match["metadata"]["outlet"],
            "source_id": match["metadata"]["source_id"],
            "similarity_score": match["score"] #cosine similarity score from 0-1
        })
    print(f"Retrieved {len(chunks)} chunks from Pinecone")
    return chunks

#2 keep top chunk per outlet
def filter_by_outlet(chunks: list[dict]) -> list[dict]:
    # for each outlet keep only the most similar chunk (highest cosine score)
    # this ensures we compare one representative article per outlet
    seen = {}
    for c in chunks:
        outlet = c["outlet"]
        if outlet not in seen:
            seen[outlet] = c  # chunks are already sorted by similarity desc
    filtered = list(seen.values())
    print(f"Filtered to {len(filtered)} outlets: {[c['outlet'] for c in filtered]}")
    return filtered

#3 TRIBE re-rank
def tribe_rerank(chunks: list[dict]) -> list[dict]:
    # send chunks to TRIBE server for engagement scoring
    texts = [c["text"] for c in chunks]
    resp = requests.post(TRIBE_URL, json={"chunks": texts})
    resp.raise_for_status()

    ranked = resp.json()["ranked_chunks"]
    score_map = {r["chunk"]: r["engagement_score"] for r in ranked}
    activation_map = {r["chunk"]: r["brain_activation"] for r in ranked}

    for c in chunks:
        c["engagement_score"] = score_map.get(c["text"], 0.0)
        c["brain_activation"] = activation_map.get(c["text"], [])

    chunks.sort(key=lambda x: x["engagement_score"], reverse=True)
    print(f"TRIBE re-ranked {len(chunks)} outlet chunks")
    return chunks

#4 generate answer w/ Claude
def generate_answer(query: str, chunks: list[dict]) -> str:
    # build context showing each outlet's coverage + engagement score
    context = "\n\n".join([
        f"[{c['outlet']} | Neural Engagement: {c['engagement_score']:.4f}]\n"
        f"Headline: {c['title']}\n"
        f"URL: {c['url']}"
        for c in chunks
    ])

    msg = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": f"""You are TechPulse, a media intelligence tool that analyzes how different news outlets cover the same story.

The following articles were retrieved from different outlets and ranked by neural engagement score — a measure of how emotionally stimulating the headline and description would be to the human brain. Higher scores indicate more emotionally charged framing.

{context}

User query: {query}

Summarize how each outlet is covering this story. Note any meaningful differences in framing or emotional tone based on their engagement scores. Be objective and analytical — do not call any outlet biased or manipulative."""
        }]
    )
    return msg.content[0].text

#full pipeline

def query_pipeline(query: str) -> dict:
    print(f"\n Query: {query}")

    chunks = retrieve(query, top_k=50)
    chunks = filter_by_outlet(chunks)
    chunks = tribe_rerank(chunks)
    answer = generate_answer(query, chunks)

    return {
        "query": query,
        "answer": answer,
        "outlet_scores": [
            {
                "outlet": c["outlet"],
                "title": c["title"],
                "url": c["url"],
                "engagement_score": c["engagement_score"],
                "similarity_score": c["similarity_score"],
                "brain_activation": c["brain_activation"]
            }
            for c in chunks
        ]
    }

