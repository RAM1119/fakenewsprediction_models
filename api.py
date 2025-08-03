from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

app = Flask(__name__)

# Global variables for models
bert_tokenizer = None
bert_model = None
distilbert_tokenizer = None
distilbert_model = None

def load_models():
    global bert_tokenizer, bert_model, distilbert_tokenizer, distilbert_model
    
    print("Loading BERT model...")
    bert_tokenizer = AutoTokenizer.from_pretrained('ramgunturu/news-detection-bert')
    bert_model = AutoModelForSequenceClassification.from_pretrained('ramgunturu/news-detection-bert')
    
    print("Loading DistilBERT model...")
    distilbert_tokenizer = AutoTokenizer.from_pretrained('ramgunturu/news-detection-distilbert')
    distilbert_model = AutoModelForSequenceClassification.from_pretrained('ramgunturu/news-detection-distilbert')
    
    print("All models loaded successfully!")

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running", "endpoints": ["/predict/bert", "/predict/distilbert"]})

@app.route('/predict/bert', methods=['POST'])
def predict_bert():
    try:
        data = request.json
        article_text = data['text']
        
        inputs = bert_tokenizer(article_text, truncation=True, padding=True, max_length=512, return_tensors='pt')
        
        with torch.no_grad():
            outputs = bert_model(**inputs)
        
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

@app.route('/predict/distilbert', methods=['POST'])
def predict_distilbert():
    try:
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
    load_models()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)