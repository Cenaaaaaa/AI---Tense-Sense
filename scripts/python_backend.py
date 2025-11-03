from flask import Flask, request, jsonify
import joblib
import re
import os

app = Flask(__name__)

# Load pre-trained model and vectorizer
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, '../public/tense_model.pkl')
vectorizer_path = os.path.join(script_dir, '../public/tfidf_vectorizer.pkl')

print(f"[v0] Loading model from {model_path}")
print(f"[v0] Loading vectorizer from {vectorizer_path}")

try:
    classifier = joblib.load(model_path)
    tfidf = joblib.load(vectorizer_path)
    print("[v0] Model and vectorizer loaded successfully!")
except FileNotFoundError as e:
    print(f"[v0] Error: Model files not found. Please run scripts/train_model.py first")
    print(f"[v0] Error details: {e}")
    classifier = None
    tfidf = None

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

tense_mapping = {
    1: "Present Tense",
    2: "Past Tense", 
    3: "Future Tense"
}

@app.route('/predict', methods=['POST'])
def predict():
    if classifier is None or tfidf is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        sentence = data.get('sentence', '').strip()
        
        if not sentence:
            return jsonify({'error': 'No sentence provided'}), 400
        
        print(f"[v0] Processing: {sentence}")
        
        # Preprocess
        cleaned = preprocess_text(sentence)
        print(f"[v0] Cleaned: {cleaned}")
        
        # Vectorize
        X = tfidf.transform([cleaned])
        
        # Predict
        prediction = classifier.predict(X)[0]
        confidence = classifier.predict_proba(X)[0]
        
        tense = tense_mapping.get(prediction, "Unknown")
        
        print(f"[v0] Prediction: {tense}, Confidence: {max(confidence):.2%}")
        
        return jsonify({
            'tense': tense,
            'prediction_value': int(prediction),
            'confidence': float(max(confidence))
        })
    
    except Exception as e:
        print(f"[v0] Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("[v0] Starting Flask backend on port 5000...")
    app.run(debug=True, port=5000)
