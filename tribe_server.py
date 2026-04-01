import multiprocessing
multiprocessing.set_start_method("fork", force=True)

import tempfile
import numpy as np
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from tribev2 import TribeModel

import torch
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

app = FastAPI()
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
print("TRIBE model loaded")

#request schema
class scoreRequest(BaseModel):
    chunks: list[str] #list of test chunks to score

#helper to score a single chunk
def score_chunk(text: str) -> float:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(text)
        tmp_path = f.name
    
    df = model.get_events_dataframe(text_path=tmp_path)

    preds, _ = model.predict(events=df)

    #average activation across all timestamps for the same neurons
    mean_activation = preds.mean(axis=0)

    #engagement score by taking mean activation across the engagement neurons
    engagement_score = float(mean_activation.mean())

    Path(tmp_path).unlink()
    return engagement_score, mean_activation.tolist() 

#endpoint
@app.post("/score")
async def score_chunks(request: scoreRequest):
    results = []
    for i,chunk in enumerate(request.chunks):
        score, activation = score_chunk(chunk)
        results.append({"chunk": chunk, "index":i, "engagement_score": score, "brain_activation": activation})
    #sort by highest engagement score
    results.sort(key=lambda x: x["engagement_score"], reverse=True)
    return {"ranked_chunks": results}

@app.get("/health")
def health():
    return {"status": "ok"}



