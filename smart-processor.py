from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch
from symspellpy import SymSpell

app = FastAPI(title="ILTC Intent Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Sentence Transformer model
MODEL_PATH = "models/iltc_finetuned_model"
model = SentenceTransformer(MODEL_PATH)

# Load the intent mapping
try:
    from functionality_map import functionality
except ImportError:
    raise ImportError("Error: functionality_map.py file is missing!")

intent_texts = list(functionality.keys())
intent_embeddings = model.encode(intent_texts, convert_to_tensor=True)

# Initialize SymSpell for spell correction
sym_spell = SymSpell(max_dictionary_edit_distance=2)
if not sym_spell.load_dictionary("ILTC_dictionary.txt", term_index=0, count_index=1):
    raise RuntimeError("Error: Could not load ILTC_dictionary.txt!")

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def root():
    return {"message": "Intent Detection API is running!"}

@app.post("/predict/")
def predict_intent(request: QueryRequest):
    user_query = request.query.strip()

    # Spell correction
    corrected_text = sym_spell.lookup_compound(user_query, max_edit_distance=2)
    corrected_query = corrected_text[0].term if corrected_text else user_query

    try:
        query_embedding = model.encode(corrected_query, convert_to_tensor=True)

        if torch.isnan(query_embedding).any():
            raise HTTPException(status_code=400, detail="Error: Embedding contains NaN values!")

        similarity_scores = util.pytorch_cos_sim(query_embedding, intent_embeddings)[0]

        threshold = 0.85
        results = [
            {"intent": intent_texts[i], "route": functionality[intent_texts[i]], "score": similarity_scores[i].item()}
            for i in range(len(intent_texts)) if similarity_scores[i].item() > threshold
        ]
        results.sort(key=lambda x: x["score"], reverse=True)

        if not results:
            return {"message": "No matching intents found above threshold", "corrected_query": corrected_query}

        return {"corrected_query": corrected_query, "results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")