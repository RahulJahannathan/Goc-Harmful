import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Data: English and Tounglish Spam vs Not Spam
data = [
    # English - Spam
    ("Congratulations! You've won a free iPhone. Click here!", "yes"),
    ("Update your account password immediately at this link.", "yes"),
    ("Claim your $1000 gift card now!", "yes"),
    ("Verify your PayPal account here", "yes"),

    # English - Not Spam
    ("Hi, how are you?", "no"),
    ("Let's meet tomorrow at the cafe", "no"),
    ("Don't forget our meeting at 3 PM", "no"),
    ("I reached home safely", "no"),

    # Tounglish - Spam
    ("Vazhthukkal! Neenga oru free iPhone win panirukeenga!", "yes"),
    ("Unga account suspend aagiruku. Link click panni update pannunga.", "yes"),
    ("Instant money kadaikkum. Inga apply pannunga!", "yes"),
    ("Neenga lucky winner! Ippo dhan chance. Miss pannadheenga!", "yes"),

    # Tounglish - Not Spam
    ("Iniku coffee ku meet pannalama?", "no"),
    ("Na safe ah reach aagiten", "no"),
    ("Kalaila class iruku. Late aagadhe", "no"),
    ("Enaku konjam help venum for the project", "no")
]

# Separate texts and labels
texts, labels = zip(*data)

# Create pipeline with TF-IDF and Naive Bayes Classifier
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Train the model
model.fit(texts, labels)

model_path = os.path.join(os.path.dirname(__file__), 'spam_model.pkl')
joblib.dump(model, model_path)
print(f"Model trained and saved at: {model_path}")
