# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch

# Initialize FastAPI
app = FastAPI()

class TextRequest(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    predicted_class: str

classes = ["non-hateful", "hateful"]

# Load pre-trained BERT model and tokenizer
MODEL_PATH = "models/camembert_mad_v1"
tokenizer = CamembertTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_PATH)
model = CamembertForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH, # Use the 12-layer CamemBERT
    num_labels=len(classes),                  # Binary classification.
    output_attentions=False,                  # Whether the model returns attentions weights.
    output_hidden_states=False,               # Whether the model returns all hidden-states.
)

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": MODEL_PATH.split("/")[1]}

# Define a route for inference
@app.post("/predict/", response_model=PredictionOutput)
async def predict(request: TextRequest):
    inputs = tokenizer(request.text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    return {"predicted_class": classes[predicted_class]}
