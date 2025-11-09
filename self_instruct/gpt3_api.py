# ==============================================================
#  gpt3_api.py — Fireworks Llama 3 Integration for Self-Instruct
#  (Google Colab compatible, uses environment variable)
# ==============================================================

import json
import time
import sys
import os
import random
import argparse
import requests
from datetime import datetime
import tqdm

# --- FIREWORKS API CONFIGURATION ---
# ✅ FIXED: do NOT put your API key directly here
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")  
API_ENDPOINT = "https://api.fireworks.ai/inference/v1/chat/completions"
MODEL_ID = "accounts/fireworks/models/llama-v3p1-8b-instruct"

def make_requests(
        engine, prompts, max_tokens, temperature, top_p,
        frequency_penalty, presence_penalty, stop_sequences,
        logprobs, n, best_of, retries=3, api_key=None, organization=None
    ):
    api_key = api_key or FIREWORKS_API_KEY
    if not api_key:
        print("FATAL ERROR: FIREWORKS_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)
        
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    if isinstance(prompts, str):
        prompts = [prompts]

    all_results = []
    
    for prompt in prompts:
        data = {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": 1
        }
        
        response = None
        for attempt in range(retries):
            try:
                response = requests.post(API_ENDPOINT, headers=headers, json=data, timeout=60)
                if response.status_code == 403:
                    print(f"403 Forbidden — Check API key permissions or model name.\nModel: {MODEL_ID}")
                    print(f"Your API key starts with: {api_key[:6]}...")
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                print(f"API Request Failed (Attempt {attempt + 1}/{retries}): {e}. Retrying in {10 * (attempt + 1)}s.", file=sys.stderr)
                time.sleep(10 * (attempt + 1))
        
        if response and response.status_code == 200:
            response_json = response.json()
            generated_text = response_json['choices'][0]['message']['content']
        else:
            generated_text = "API_GENERATION_FAILED"

        result = {
            "prompt": prompt,
            "response": {"choices": [{"text": generated_text, "finish_reason": "stop"}]},
            "created_at": str(datetime.now()),
        }
        all_results.append(result)
        
    return all_results
