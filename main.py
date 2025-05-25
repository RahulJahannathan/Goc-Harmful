from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

# Load the model safely
model_path = "spam_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = joblib.load(model_path)

# Define request schema
class CommentRequest(BaseModel):
    comment: str

# Define response schema (optional, improves docs)
class PredictionResponse(BaseModel):
    comment: str
    prediction: str

app = FastAPI()

def predict_spam(text: str) -> str:
    try:
        prediction = model.predict([text])
        return prediction[0]
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")

@app.post("/predict/", response_model=PredictionResponse)
async def predict(request: CommentRequest) -> PredictionResponse:
    if not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment is required")

    try:
        result = predict_spam(request.comment)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictionResponse(
        comment=request.comment,
        prediction="Spam" if result == "yes" else "Not Spam"
    )
