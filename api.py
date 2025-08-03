from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

app = Flask(__name__)

# Global variables for DistilBERT only
distilbert_tokenizer = None
distilbert_model = None

def load_distilbert_if_needed():
    global distilbert_tokenizer, distilbert_model
    if distilbert_tokenizer is None:
        print("Loading DistilBERT model...")
        distilbert_tokenizer = AutoTokenizer.from_pretrained('ramgunturu/news-detection-distilbert')
        distilbert_model = AutoModelForSequenceClassification.from_pretrained('ramgunturu/news-detection-distilbert')
        print("DistilBERT loaded successfully!")

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running", "model": "DistilBERT", "endpoint": "/predict"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        load_distilbert_if_needed()
        
        data = request.json
        article_text = data['text']
        
        inputs = distilbert_tokenizer(article_text, truncation=True, padding=True, max_length=512, return_tensors='pt')
        
        with torch.no_grad():
            outputs = distilbert_model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        prediction = torch.argmax(outputs.logits, dim=-1)[0]
        
        return jsonify({
            'prediction': 'Reliable' if prediction == 1 else 'Unreliable',
            'confidence': float(probs[prediction] * 100),
            'scores': {
                'unreliable': float(probs[0] * 100),
                'reliable': float(probs[1] * 100)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
