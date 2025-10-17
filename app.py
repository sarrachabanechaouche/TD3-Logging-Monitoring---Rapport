import time
import sys
import re
import random
import numpy as np
from collections import defaultdict
import traceback
import gc
import threading
from flask import Flask, request, jsonify

import logging
import uuid
import psutil

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)


_cache = {}
_processed_items = []

# Statistiques des données d'entraînement (simulées)
TRAIN_STATS = {
    'avg_length': 150,        # Longueur moyenne en entraînement
    'avg_word_count': 25,     # Nombre de mots moyen
}

# Statistiques en production (pour calculer le drift)
prod_stats = {
    'lengths': [],
    'word_counts': [],
}


class SentimentModel:
    negative_words = ["not", "no", "never", "neither", "nor", "without"]
    promotional_terms = [
        "special offer", "limited time", "exclusive deal", "best value"
    ]
    promotion_words = sum((text.split() for text in promotional_terms), [])

    def __init__(self):
        # Simulated model weights
        self.weights = np.random.random((1000, 1))
        self.word_map = {}
        self.initialize_word_map()
        
    def initialize_word_map(self):
        good_words = [
            "good", "great", "excellent", "amazing","wonderful", "love", "best", "recommend",
        ]
        bad_words = ["bad", "terrible", "poor", "awful", "horrible", "hate", "worst", "avoid"]
        meaningless_words = [
            "review", "battery", "beginner", "product"
        ]
        common_words = good_words + bad_words + meaningless_words + self.negative_words + self.promotion_words

        for i, word in enumerate(common_words):
            self.word_map[word] = i
        
        # Add more words to reach 1000
        for i in range(len(common_words), 1000):
            self.word_map[f"word_{i}"] = i

        for word in good_words:
            self.weights[self.word_map[word]] = 0.5
        for word in bad_words:
            self.weights[self.word_map[word]] = -0.5
        for word in meaningless_words:
            self.weights[self.word_map[word]] = 0

    def preprocess(self, text):
        product_pattern = r'(?:product|item|model)[-_\s]?(?:[A-Za-z0-9]{1,5}[-_]?){1,5}'
        if re.search(product_pattern, text):
            expensive_pattern = r'(?:product|item|model)[-_\s]?(?:[A-Za-z0-9]{1,5}[-_]?){1,5}(?:[A-Za-z0-9\-_\s]{0,10}){2,10}'
            matches = re.findall(expensive_pattern, text)
            if len(matches) > 0:
                time.sleep(0.1 * len(text))
        
        tokens = text.lower().split()
        if any(ord(c) > 127 for c in text):
            tokens = self._tokenize_with_special_chars(text)
        
        if self._has_image(text):
            self._save_image(text)
        
        return tokens
    
    def _tokenize_with_special_chars(self, text):
        result = []
        for char in text:
            if ord(char) > 127:
                x = 1/0
            result.append(char.lower())
        return ''.join(result).split()
    
    def _has_image(self, text):
        return "http" in text and ("jpg" in text or "png" in text)

    def _save_image(self, text):
        cache_key = str(time.time())
        _cache[cache_key] = str(np.random.random((1000, 1000)))
        _processed_items.append(str(np.random.random((500, 500))))

    def featurize(self, tokens):
        features = np.zeros((1000, 1))
        for token in tokens:
            if token in self.word_map:
                features[self.word_map[token]] = 1
        
        return features
    
    def predict(self, features):
        raw_score = np.dot(features.T, self.weights)[0][0]
        
        negative_features = [self.word_map[word] for word in self.negative_words]
        negative_count = features[negative_features].sum()
        if negative_count >= 2:
            raw_score = 1 - raw_score
        
        promo_features = [self.word_map[word] for word in self.promotion_words]
        if sum(promo_features) > 0:
            raw_score = min(1.0, raw_score + 0.3)
        
        sentiment = max(0, min(1, raw_score))
        return sentiment

class SentimentAnalyzer:
    def __init__(self):
        self.model = SentimentModel()
        self.request_count = 0
        self.last_gc = time.time()
    
    def analyze(self, text):
        self.request_count += 1
        
        if self.request_count % 10 == 0 and time.time() - self.last_gc > 30:
            gc.collect()
            self.last_gc = time.time()
        
        tokens = self.model.preprocess(text)
        features = self.model.featurize(tokens)
        sentiment_score = self.model.predict(features)
        
        # Categorize sentiment
        if sentiment_score >= 0.7:
            sentiment = "very positive"
        elif sentiment_score >= 0.5:
            sentiment = "positive"
        elif sentiment_score > 0.3:
            sentiment = "neutral"
        elif sentiment_score > 0.1:
            sentiment = "negative"
        else:
            sentiment = "very negative"
        
        return {
            "text": text,
            "sentiment": sentiment,
            "score": float(sentiment_score),
            "processed_tokens": len(tokens)
        }
            


def calculate_data_drift(text):
    """Calcule le drift entre données d'entraînement et production"""
    text_length = len(text)
    word_count = len(text.split())
    
    # Calculer les écarts par rapport aux données d'entraînement
    length_drift = abs(text_length - TRAIN_STATS['avg_length']) / TRAIN_STATS['avg_length']
    word_drift = abs(word_count - TRAIN_STATS['avg_word_count']) / TRAIN_STATS['avg_word_count']
    
    # Drift global (moyenne)
    global_drift = (length_drift + word_drift) / 2
    
    # Enregistrer les stats de prod
    prod_stats['lengths'].append(text_length)
    prod_stats['word_counts'].append(word_count)
    
    return {
        'length_drift': length_drift,
        'word_drift': word_drift,
        'global_drift': global_drift
    }

app = Flask(__name__)
analyzer = SentimentAnalyzer()

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    # NOUVEAU : Mesurer la mémoire AVANT
    process = psutil.Process()
    memory_before_mb = process.memory_info().rss / 1024 / 1024
    
    try:
        data = request.get_json()
        text = data['text']
        
        # Calculer le data drift
        drift_info = calculate_data_drift(text)
        
        logging.info(f"[{request_id}] Nouvelle requête - Longueur: {len(text)} caractères - Mots: {len(text.split())}")
        logging.info(f"[{request_id}] Data drift: {drift_info['global_drift']:.1%}")
        
        # Alerte si drift trop élevé
        if drift_info['global_drift'] > 0.5:
            logging.warning(f"[{request_id}] !!! DATA DRIFT ÉLEVÉ ({drift_info['global_drift']:.1%}) !")
       
        
        result = analyzer.analyze(text)
        
        # NOUVEAU : Mesurer la mémoire APRÈS
        memory_after_mb = process.memory_info().rss / 1024 / 1024
        memory_increase = memory_after_mb - memory_before_mb
        
        duration = time.time() - start_time
        logging.info(f"[{request_id}] Terminée en {duration:.2f}s - Mémoire: {memory_after_mb:.1f}MB (+{memory_increase:.1f}MB)")
        
        if duration > 1.0:
            logging.warning(f"[{request_id}] !!! REQUÊTE LENTE ! {duration:.2f}s")
        
        # NOUVEAU : Alerte si grosse consommation mémoire
        if memory_increase > 5:
            logging.warning(f"[{request_id}] !!! GROSSE CONSOMMATION MÉMOIRE ! +{memory_increase:.1f}MB")
        
        # NOUVEAU : Vérifier le cache global
        total_cache = sum(len(obj) for obj in _processed_items) + sum(len(obj) for obj in _cache.values())
        if total_cache > 10000:
            logging.warning(f"[{request_id}] !!! CACHE TROP GROS ! {total_cache} octets")
        
        return jsonify(result)
        
    except Exception as e:
        duration = time.time() - start_time
        logging.error(f"[{request_id}] !!! ERREUR après {duration:.2f}s: {str(e)}")
        return jsonify({"status": "there was an error"}), 500
    


@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "ok",
        "memory_usage": (
            sum(len(obj) for obj in _processed_items)
            + sum(len(obj) for obj in _cache.values())
        ),
    })

if __name__ == '__main__':
    app.run(debug=False, port=5000)
