import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import json
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
model_output_path = os.path.join(script_dir, '../public/model_data.json')

print("[v0] Loading dataset from:", csv_path)
try:
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')
    print(f"[v0] Loaded {len(df)} samples")
except Exception as e:
    print(f"[v0] Error loading CSV: {e}")
    exit(1)

# Preprocess sentences
print("[v0] Preprocessing text...")
df['cleaned_sentence'] = df['Sentence'].apply(preprocess_text)

# Prepare features and labels
X = df['cleaned_sentence']
y = df['Label']

print(f"[v0] Label distribution:")
print(y.value_counts().sort_index())

# Create TF-IDF vectorizer
print("[v0] Training TF-IDF vectorizer...")
tfidf = TfidfVectorizer(
    max_features=1000, 
    ngram_range=(1, 2), 
    min_df=2, 
    max_df=0.95,
    stop_words='english'
)
X_tfidf = tfidf.fit_transform(X)

# Train Logistic Regression
print("[v0] Training Logistic Regression classifier...")
classifier = LogisticRegression(
    max_iter=1000, 
    solver='lbfgs', 
    multi_class='multinomial', 
    random_state=42,
    C=1.0
)
classifier.fit(X_tfidf, y)

# Evaluate
train_accuracy = classifier.score(X_tfidf, y)
print(f"[v0] Training accuracy: {train_accuracy*100:.2f}%")

# Export model data to JSON
print("[v0] Exporting model data to JSON...")

model_data = {
    "vocabulary": {word: idx for word, idx in tfidf.vocabulary_.items()},
    "idf": tfidf.idf_.tolist(),
    "coefficients": classifier.coef_.tolist(),
    "intercept": classifier.intercept_.tolist(),
    "classes": classifier.classes_.tolist(),
    "tfidf_params": {
        "max_features": tfidf.max_features,
        "ngram_range": list(tfidf.ngram_range),
        "min_df": tfidf.min_df,
        "max_df": tfidf.max_df
    }
}

with open(model_output_path, 'w') as f:
    json.dump(model_data, f)

print(f"[v0] Model exported to: {model_output_path}")
print(f"[v0] Vocabulary size: {len(model_data['vocabulary'])}")
print(f"[v0] Model parameters exported successfully!")

# Test predictions
print("\n[v0] Testing predictions...")
test_sentences = [
    "I am playing soccer",
    "They went to the movies yesterday", 
    "She will travel to Paris next week",
    "The sun is shining bright",
    "He was reading a book",
    "We will have finished by then"
]

tense_mapping = {1: "Present Tense", 2: "Past Tense", 3: "Future Tense"}

for sentence in test_sentences:
    cleaned = preprocess_text(sentence)
    vectorized = tfidf.transform([cleaned])
    prediction = classifier.predict(vectorized)[0]
    proba = classifier.predict_proba(vectorized)[0]
    max_proba = max(proba)
    print(f"[v0] '{sentence}' -> {tense_mapping[prediction]} (confidence: {max_proba*100:.1f}%)")
