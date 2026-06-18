# File name: agri_rag_demo_custom_api.py

import faiss
import json
import http.client
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# ======================
# 1. Custom LLM call function
# ======================
def call_with_messages(api_key, prompt, temperature=0.7, max_n_tokens=500, top_p=1.0, model_name="gpt-3.5-turbo"):
    # Use low-level http.client to call (pseudo) OpenAI-style Chat Completions API and return generated text
    conn = http.client.HTTPSConnection("api.openai-hub.com")
    payload = json.dumps({
       "model": model_name,
       "messages": [
          {
             "role": "user", # Specify message role as user
             "temperature": temperature, # Pass temperature (placed inside message object as per original code)
             "top_p": top_p, # Pass top_p
             "max_tokens": max_n_tokens, # Pass max_tokens
             "content": prompt  # Message content (place the entire prompt in the content field)
          }
       ]
    })
    # Construct request headers
    headers = {
       'Authorization': api_key,
       'User-Agent': 'Apifox/1.0.0 (https://apifox.com)', # Custom User-Agent for server-side request identification
       'Content-Type': 'application/json' # Declare request body as JSON
    }
    # Send POST request to the specified path /v1/chat/completions
    conn.request("POST", "/v1/chat/completions", payload, headers)
    # Get response object
    res = conn.getresponse()
    # Read response body bytes
    data = res.read()
    # Decode response bytes as UTF-8 and parse as JSON object
    response_json = json.loads(data.decode("utf-8"))
    # Simple error check: if the returned JSON does not contain the "choices" field, treat as request failure
    if "choices" not in response_json:
        return "Request failed!"
    else:
        # According to the example structure, return the message.content field from the first choice
        # Note: Different APIs may have different response structures (e.g., OpenAI Chat Completions returns choices[0].message.content)
        # Here we follow the original code's extraction approach; callers should adjust based on the actual API response structure
        return response_json["choices"][0]["message"]["content"]