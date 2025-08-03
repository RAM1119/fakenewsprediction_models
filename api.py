from flask import Flask, request, jsonify
import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Force minimal memory usage
torch.set_num_threads(1)
torch.backends.cudnn.enabled = False
os.environ['CUDA_VISIBLE_DEVICES'] = ''

app = Flask(__name__)

# Global variables
distilbert_tokenizer = None
distilbert_model = None

def load_distilbert_optimized():
    global distilbert_tokenizer, distilbert_model
    if distilbert_tokenizer is None:
        print("Loading optimized DistilBERT...")
        
        # Clear memory first
        gc.collect()
        
        try:
            # Load tokenizer with reduced max length
            distilbert_tokenizer = AutoTokenizer.from_pretrained(
                'ramgunturu/news-detection-distilbert',
                model_max_length=256,
                use_fast=True
            )
            
            # Load model with memory optimizations
            distilbert_model = AutoModelForSequenceClassification.from_pretrained(
                'ramgunturu/news-detection-distilbert',
                torch_dtype=torch.float16,  # Half precision
                low_cpu_mem_usage=True,
                device_map="cpu"
            )
            
            # Optimize for inference
            distilbert_model.eval()
            
            # Clear memory after loading
            gc.collect()
            
            print("DistilBERT loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to basic loading
            distilbert_tokenizer = AutoTokenizer.from_pretrained('ramgunturu/news-detection-distilbert')
            distilbert_model = AutoModelForSequenceClassification.from_pretrained('ramgunturu/news-detection-distilbert')
            print("DistilBERT loaded with basic settings!")

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "API is running", 
        "model": "DistilBERT (Optimized)", 
        "endpoint": "/predict",
        "memory_usage": "Optimized for 512MB"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load model if not loaded
        load_distilbert_optimized()
        
        # Get input data
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
            
        article_text = data['text']
        
        # Tokenize with memory optimization
        inputs = distilbert_tokenizer(
            article_text,
            truncation=True,
            padding=True,
            max_length=256,  # Reduced from 512
            return_tensors='pt'
        )
        
        # Predict with memory management
        with torch.no_grad():
            outputs = distilbert_model(**inputs)
        
        # Process results
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        prediction = torch.argmax(outputs.logits, dim=-1)[0]
        
        # Clear intermediate tensors
        del inputs, outputs
        gc.collect()
        
        # Return results
        result = {
            'prediction': 'Reliable' if prediction == 1 else 'Unreliable',
            'confidence': float(probs[prediction] * 100),
            'scores': {
                'unreliable': float(probs[0] * 100),
                'reliable': float(probs[1] * 100)
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    try:
        # Check if model is loaded
        model_loaded = distilbert_model is not None
        return jsonify({
            "status": "healthy",
            "model_loaded": model_loaded,
            "memory_optimized": True
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting optimized server on port {port}")
    print("Memory optimizations enabled")
    
    # Don't load models at startup - load on first request
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
