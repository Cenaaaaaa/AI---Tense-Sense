import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import re
import os

def preprocess_text(text):
    """Preprocess text by lowercasing and removing special characters"""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# Set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '../public/dataset.csv')
model_path = os.path.join(script_dir, '../public/tense_model.pkl')
vectorizer_path = os.path.join(script_dir, '../public/tfidf_vectorizer.pkl')

print("[v0] Loading dataset...")
try:
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')
    print(f"[v0] Loaded {len(df)} samples")
    print(f"[v0] Columns: {df.columns.tolist()}")
    print(f"[v0] Sample data:\n{df.head()}")
except Exception as e:
    print(f"[v0] Error loading CSV: {e}")
    exit(1)

# Preprocess sentences
print("[v0] Preprocessing text...")
df['cleaned_sentence'] = df['Sentence'].apply(preprocess_text)
print(f"[v0] Sample cleaned sentences:\n{df[['Sentence', 'cleaned_sentence', 'Label']].head()}")

# Prepare features and labels
X = df['cleaned_sentence']
y = df['Label']

print(f"[v0] Label distribution:")
print(y.value_counts().sort_index())

# Create TF-IDF vectorizer and Logistic Regression pipeline
print("[v0] Training model with TF-IDF + Logistic Regression...")
tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=2, max_df=0.9)
X_tfidf = tfidf.fit_transform(X)

classifier = LogisticRegression(max_iter=500, solver='lbfgs', multi_class='multinomial', random_state=42)
classifier.fit(X_tfidf, y)

# Evaluate on full dataset
train_accuracy = classifier.score(X_tfidf, y)
print(f"[v0] Training accuracy: {train_accuracy*100:.2f}%")

# Save model and vectorizer
print("[v0] Saving model and vectorizer...")
joblib.dump(tfidf, vectorizer_path)
joblib.dump(classifier, model_path)
print(f"[v0] Model saved to {model_path}")
print(f"[v0] Vectorizer saved to {vectorizer_path}")

# Test predictions
print("\n[v0] Testing predictions...")
test_sentences = [
    "I am playing soccer",
    "They went to the movies yesterday",
    "She will travel to Paris next week"
]

tense_mapping = {1: "Present Tense", 2: "Past Tense", 3: "Future Tense"}

for sentence in test_sentences:
    cleaned = preprocess_text(sentence)
    vectorized = tfidf.transform([cleaned])
    prediction = classifier.predict(vectorized)[0]
    print(f"[v0] '{sentence}' -> {tense_mapping[prediction]}")
