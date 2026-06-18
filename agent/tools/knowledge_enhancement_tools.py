# File name: agri_rag_demo_custom_api.py
import os

import faiss
import json
import http.client
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from agent.llm import qwen_vl_max
from agent.tools.browser_search import retrieval_augmented_generation


# ======================
# 1. Custom LLM call function
# ======================
def call_with_messages(api_key, prompt, temperature=0.7, max_n_tokens=500, top_p=1.0, model_name="gpt-3.5-turbo"):
    """
        Call LLM using custom API and return generated result
        """
    # Create HTTPS connection
    conn = http.client.HTTPSConnection("api.openai-hub.com")
    # Construct request payload in JSON format
    payload = json.dumps({
       "model": model_name,
       "messages": [
          {
             "role": "user",
             "temperature": temperature,
             "top_p": top_p,
             "max_tokens": max_n_tokens,
             "content": prompt
          }
       ]
    })
    # Set request headers
    headers = {
       'Authorization': api_key,
       'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
       'Content-Type': 'application/json'
    }
    # Send POST request to LLM API
    conn.request("POST", "/v1/chat/completions", payload, headers)
    # Get response
    res = conn.getresponse()
    data = res.read()
    # Parse response data as JSON
    response_json = json.loads(data.decode("utf-8"))
    # Check if result is returned
    if "choices" not in response_json:
        return "Request failed!"
    else:
        # Return LLM output text
        return response_json["choices"][0]["message"]["content"]


# ======================
# 4. Retrieval function
# ======================
def retrieve_documents(query, vectorizer, faiss_index, knowledge_texts, top_k=1):
    # Use TF-IDF + Faiss to retrieve the most relevant knowledge documents
    # Vectorize the query text
    query_vec = vectorizer.transform([query]).toarray()
    # Search for top_k nearest vectors in Faiss index
    distances, indices = faiss_index.search(query_vec.astype(np.float32), top_k)
    # Return original texts based on retrieved indices
    results = [knowledge_texts[i] for i in indices[0]]
    return results

# ======================
# 5. Build RAG query
# ======================
def knowledge_enhanced_answer(crop, disease, api_key, knowledge_texts, vectorizer, faiss_index, oss_url, vl_api_key, is_image_model):
    # Combine crop and disease information, retrieve from knowledge base and generate knowledge-enhanced answer
    # Construct query text
    query = f"Crop: {crop}\nMost probable disease: {disease}"
    # Retrieve relevant documents
    docs = retrieve_documents(query, vectorizer, faiss_index, knowledge_texts)
    # Merge retrieved documents into context
    context = "\n".join(docs)
    # print(context)
    # Construct LLM prompt
    # prompt = f"{oss_url}, This is my crop pest and disease image. Please refer to this image and the content below to generate a report in Chinese: {context}"
    # Call image analysis model to analyze pest and disease image
    # Whether to call image analysis
    if is_image_model is not None and vl_api_key is not None and oss_url:
        prompt = f"{oss_url}, analyze this image"
        image_analyze = qwen_vl_max.call_with_messages(vl_api_key, prompt, temperature=1, max_n_tokens=50, top_p=1)
    else:
        image_analyze = "Image analysis not performed"
    # Call LLM to summarize and generate report
    new_prompt = f"Please refer to the image analysis and the knowledge base content below to generate a report in Chinese. Do not include content about walnut root rot in the report: (1) Image analysis: {image_analyze} (2) Knowledge base content: {context}"
    #print(f"{context}")
    # new_prompt = f"Please refer to the content below to generate a report in Chinese: {context}"
    answer = call_with_messages(api_key, new_prompt, temperature=0.5, max_n_tokens=500, top_p=0.1)
    return answer


def get_knowledge_by_crop_disease(crop: str, disease: str, json_file):
    """
    Read the report content from the JSON file for the first entry that contains the specified crop and disease.

    Parameters:
        crop (str): Crop name, e.g., 'Rice'
        disease (str): Disease name, e.g., 'Rice Blast'
        json_file (str): JSON file path, default is 'data.json'

    Returns:
        str: If a matching report is found, return the report content (assigned to knowledge_texts)
             If not found, return None
    """
    if not os.path.isfile(json_file):
        print(f"⚠️ File does not exist: {json_file}")
        return None

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"⚠️ JSON file format error: {json_file}")
        return None

    # Iterate through each record in the JSON data
    for entry in data:
        report = entry.get('report', '')
        # Check if the report contains both crop and disease
        if crop in report and disease in report:
            knowledge_texts = report
            print(f"✅ Matching report found, crop: {crop} | disease: {disease}")
            return knowledge_texts

    print(f"❌ No report found containing '{crop}' and '{disease}'")
    return None



# Report generation function
def report_generation(crop, disease, api_key, is_google, oss_url, vl_api_key, is_image_model, memory_file):
    # Generate knowledge-enhanced report and save locally
    # Call retrieval function to get knowledge text
    if is_google:
        knowledge_texts = [retrieval_augmented_generation(f"{crop} {disease}")]
    else:
        knowledge_texts = get_knowledge_by_crop_disease(crop, disease, memory_file)
        if knowledge_texts is None:
            knowledge_texts = call_with_messages(api_key, f"Please provide control measures for the crop pest and disease: {disease} of {crop}", temperature=0, max_n_tokens=500)
    # ======================
    # 3. Build TF-IDF vector database (simplest example)
    # ======================
    vectorizer = TfidfVectorizer()# Initialize TF-IDF vectorizer
    X = vectorizer.fit_transform(knowledge_texts).toarray()# Vectorize knowledge texts
    faiss_index = faiss.IndexFlatL2(X.shape[1]) # Initialize L2 distance index
    faiss_index.add(X)# Add vectors to index
    # Generate answer using knowledge enhancement
    result = knowledge_enhanced_answer(crop, disease, api_key, knowledge_texts, vectorizer, faiss_index, oss_url, vl_api_key, is_image_model)
    # Ensure folder exists
    folder_path = r"D:\agriculture-Agent\dataset\report"
    os.makedirs(folder_path, exist_ok=True)

    # Generate filename (customizable as needed)
    file_name = f"{crop}_{disease}_.txt"
    file_path = os.path.join(folder_path, file_name)

    # Write to file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(result)
    return result

# ======================
#  Test
# ======================
if __name__ == "__main__":
    API_KEY = ""
    crop = "Tomato"
    disease = "Bacterial spot"

    # 2. Build knowledge base (example texts)
    knowledge_texts = [retrieval_augmented_generation(f"{crop} {disease}")]
    print(knowledge_texts)
    # ======================
    # 3. Build TF-IDF vector database (simplest example)
    # ======================
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(knowledge_texts).toarray()
    faiss_index = faiss.IndexFlatL2(X.shape[1])
    faiss_index.add(X)
    # Generate knowledge-enhanced answer
    result = knowledge_enhanced_answer(crop, disease, API_KEY, knowledge_texts, vectorizer, faiss_index)
    print("==== Knowledge-Enhanced Answer ====")
    print(result)