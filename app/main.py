""""""
import uvicorn
from fastapi import FastAPI, BackgroundTasks 
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import logging
import pandas as pd
from .data_preprocessing import get_data_loaders  # Your preprocessing module
from .model_training import train_model  # Your training function

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "best_model.pth"
NUM_LABELS = 5  
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = None  # Placeholder for trained model

@app.post("/train")
async def train_api(background_tasks: BackgroundTasks, file_path: str):
    """
    API endpoint to trigger model training.
    """
    def train_and_save():
        global model
        train_loader, val_loader, categories = get_data_loaders(file_path, tokenizer)
        model = train_model(train_loader, val_loader, num_labels=len(categories))
        torch.save(model.state_dict(), MODEL_PATH)
        logger.info("Training complete. Model saved.")

    background_tasks.add_task(train_and_save)
    return {"message": "Training started. Check logs for progress."}

@app.post("/predict")
async def predict(text: str):
    """
    API endpoint to classify user input text.
    """
    global model
    if model is None:
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=NUM_LABELS)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()

    return {"text": text, "predicted_category": predicted_class}


# if __name__ == "__main__":
#     uvicorn.run(app , port =1025 ,host = "127.0.0.2")
