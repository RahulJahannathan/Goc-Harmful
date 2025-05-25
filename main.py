from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Load the model
model_path = "spam_model.pkl"
model = joblib.load(model_path)

# Define the request schema
class CommentRequest(BaseModel):
    comment: str

# Initialize FastAPI app
app = FastAPI()

# Prediction function
def predict_spam(text: str):
    prediction = model.predict([text])
    return prediction[0]

# API route
@app.post("/predict/")
def predict(request: CommentRequest):
    if not request.comment:
        raise HTTPException(status_code=400, detail="Comment is required")
    
    result = predict_spam(request.comment)
    return {
        "comment": request.comment,
        "prediction": "Spam" if result == "yes" else "Not Spam"
    }
