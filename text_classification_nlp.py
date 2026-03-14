# text_classifier.py
# NLP Text Classification using TF-IDF + Logistic Regression

import pandas as pd
import re
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# ----------------------------
# Text Preprocessing
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text


# ----------------------------
# Load Dataset
# ----------------------------
# CSV format: text,label
# Example:
# "I love this product",positive
# "Bad experience",negative

data = pd.read_csv("dataset.csv")

data["text"] = data["text"].apply(clean_text)


# ----------------------------
# Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["label"],
    test_size=0.2,
    random_state=42
)


# ----------------------------
# Build Pipeline
# ----------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )),
    ("clf", LogisticRegression(max_iter=1000))
])


# ----------------------------
# Train Model
# ----------------------------
print("Training model...")
model.fit(X_train, y_train)


# ----------------------------
# Evaluate Model
# ----------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ----------------------------
# Save Model
# ----------------------------
joblib.dump(model, "text_classifier.pkl")
print("\nModel saved as text_classifier.pkl")


# ----------------------------
# Prediction Function
# ----------------------------
def predict_text(text):
    text = clean_text(text)
    return model.predict([text])[0]


# ----------------------------
# Test Prediction
# ----------------------------
if __name__ == "__main__":

    sample_text = "This product is amazing and works perfectly"
    result = predict_text(sample_text)

    print("\nSample Text:", sample_text)
    print("Predicted Label:", result)