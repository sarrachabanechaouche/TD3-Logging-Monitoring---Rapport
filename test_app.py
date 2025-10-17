# test_app.py
import requests
import time
import concurrent.futures as concurrent_futures
import numpy as np
import json
import os

# Create results directory
if not os.path.exists('results'):
    os.makedirs('results')

BASE_URL = "http://localhost:5000"

def make_request(text):
    start_time = time.time()
    try:
        response = requests.post(
            f"{BASE_URL}/analyze",
            json={"text": text},
            timeout=30
        )
        response_data = response.json()
        elapsed = time.time() - start_time
        return {
            "success": True,
            "response": response_data,
            "status_code": response.status_code,
            "time": elapsed
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "error": str(e),
            "time": elapsed
        }

def run_test_case(test_name, texts, concurrent=False):
    results = []
    if concurrent:
        with concurrent_futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, text) for text in texts]
            for future in concurrent_futures.as_completed(futures):
                results.append(future.result())
    else:
        for text in texts:
            result = make_request(text)
            results.append(result)
    
    return results

def test_normal_cases():
    texts = [
        "I really enjoyed this product. It works exactly as described.",
        "This was a terrible purchase. Broke after one use.",
        "The product is okay, not great but not bad either.",
        "Amazing quality and fast shipping. Would buy again!",
        "I'm disappointed with the build quality."
    ]

    results = run_test_case("Normal Cases", texts)
    status_codes = [r["status_code"] for r in results]

    assert status_codes == [200] * len(texts)

def test_case_1():
    texts = [
        "The product-A123B456 has really good battery life.",
        "I bought the model X100_Z200 and it's fantastic.",
        "The product-A987C654D321B is the best model_X12345_Z67890 I've ever used in my life."
    ]
    results = run_test_case("Normal Cases", texts)

    avg_time = sum(r["time"] for r in results) / len(results)
    
    assert avg_time < 1.

def test_case_2():
    texts = [
        "I don't not like this product.",
        "There's never not a reason to buy this.",
        "I can't say I never enjoyed using this product.",
        "It's not that I wouldn't recommend this."
    ]
    results = run_test_case("Normal Cases", texts)

    assert results[0]["response"]["score"] < .5
    assert results[1]["response"]["score"] < .5
    assert results[2]["response"]["score"] > .5
    assert results[3]["response"]["score"] < .5

def test_case_3():
    texts = [
        "I love this café's products!",
        "This product is très magnifique.",
        "It's a great choice for beginners. Señor García recommends it too.",
        "This is the best naïve implementation I've seen."
    ]
    results = run_test_case("Normal Cases", texts)
    status_codes = [r["status_code"] for r in results]

    assert status_codes == [200] * len(texts)

def test_case_4():
    texts = [
        "This is a mediocre product with a special offer.",
        "The quality is average but it's a limited time deal.",
        "Not the best design but an exclusive deal makes it worth it.",
        "Would have returned it, but the best value I could find."
    ]
    results = run_test_case("Normal Cases", texts)

    assert results[0]["response"]["score"] < .5
    assert results[1]["response"]["score"] < .5
    assert results[2]["response"]["score"] < .5
    assert results[3]["response"]["score"] < .5

def test_case_5():
    texts = [
        "See my experience here: http://example.com/image1.jpg",
        "I uploaded a photo: http://mysite.com/review/product.png",
        "Check out these pics: http://pics.com/1.jpg and http://pics.com/2.png",
        "Documentation: http://manual.pdf and photos: http://gallery.com/view.jpg"
    ]
    results = run_test_case("Normal Cases", texts)
    status_codes = [r["status_code"] for r in results]

def test_case_6():
    texts = ["Short review number " + str(i) for i in range(20)]
    results = run_test_case("Normal Cases", texts)
    status_codes = [r["status_code"] for r in results]

    assert status_codes == [200] * len(texts)

def test_case_7():
    texts = [
        "Le meilleur objet que j'ai jamais acheté.",
        "Depuis que j'ai le modèle AX412, je suis heureux",
        "Je n'aime pas ce produit",
        "Aussi dégueu que le café de l'ESGI",
    ]
    results = run_test_case("Normal Cases", texts)

    assert results[0]["response"]["score"] > .5
    assert results[1]["response"]["score"] > .5
    assert results[2]["response"]["score"] < .5
    assert results[3]["response"]["score"] < .5

def tst_check_health():
    """Check the health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    res = response.json()

    assert res["memory_usage"] < 5000
