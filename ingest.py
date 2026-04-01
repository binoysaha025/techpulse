import os 
import uuid
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import requests

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("techpulse")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

TARGET_OUTLETS = {
    "bbc-news": "BBC",
    "reuters": "Reuters",
    "cnn": "CNN",
    "fox-news": "Fox News",
    "associated-press": "AP",
    "al-jazeera-english": "Al Jazeera",
}

#fetch
def fetch_news(page_size=100) -> list[dict]:
    source_ids = ",".join(TARGET_OUTLETS.keys())
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "Iran war",
        "sources": source_ids,
        "pageSize": page_size,
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY,
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])
    print(f"Fetched {len(articles)} articles from NewsAPI")
    return articles

#chunking
def chunk_articles(articles: list[dict], chunk_size=200) -> list[dict]:
    chunks = []
    for a in articles:
        title = a.get("title") or ""
        description = a.get("description") or ""
        source_id = a.get("source", {}).get("id", "unknown")
        source_name = a.get("source", {}).get("name", "Unknown")

        if not title:
            continue

        # combine title + description for richer TRIBE signal
        text = f"{title}. {description}".strip(". ") if description else title

        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "text": text,
            "title": title,
            "description": description,
            "url": a.get("url", ""),
            "outlet": source_name,
            "source_id": source_id,
            "published_at": a.get("publishedAt", ""),
        })

    print(f"Created {len(chunks)} chunks")
    return chunks

#embed and upsert
def embed_and_upsert(chunks:list[dict], batch_size=50):
    index.delete(delete_all=True)
    print("Cleared existing Pinecone vectors")

    texts = [c["text"] for c in chunks]
    embedder.encode(texts, show_progress_bar=True)
    embeddings = embedder.encode(texts)

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]

        vectors = []
        for chunk, emb in zip(batch_chunks, batch_embeddings):
            vectors.append({
                "id": chunk["chunk_id"],
                "values": emb.tolist(),
                "metadata": {
                    "text": chunk["text"],
                    "title": chunk["title"],
                    "url": chunk["url"],
                    "outlet": chunk["outlet"],
                    "source_id": chunk["source_id"],
                    "published_at": chunk["published_at"],
                }
            })
        index.upsert(vectors=vectors)
        print(f"Upserted batch {i//batch_size + 1}")

    print(f"Done — {len(chunks)} chunks in Pinecone")

#main 
if __name__ == "__main__":
    articles = fetch_news(page_size=100)
    chunks = chunk_articles(articles)
    embed_and_upsert(chunks)