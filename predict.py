import joblib

model_path = "spam_model.pkl"
model = joblib.load(model_path)

def predict_spam(text):
    prediction = model.predict([text])
    return prediction[0]

if __name__ == "__main__":
    user_input = input("Enter a comment: ")
    result = predict_spam(user_input)
    print(f"Prediction: {'Spam' if result == 'yes' else 'Not Spam'}")
