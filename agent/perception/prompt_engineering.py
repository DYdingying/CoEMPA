# Read questions_and_images JSON file
import json
import os
from datetime import datetime
import random
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer

import numpy as np

from agent.llm import qwen_vl_max
from agent.llm.gpt_4o_mini import call_with_messages
from agent.tools import aliyun_oss
from sklearn.metrics.pairwise import cosine_similarity


# Used to read questions and images from JSON file, and generate image descriptions via LLM
# Function: Read questions and image paths from JSON file
def read_questions_and_images(json_path):
    """
        Read questions and image paths from JSON file, and filter valid entries.

        Parameters:
            json_path: str, JSON file path containing a list with question, image, answer

        Returns:
            results: list[dict], each dict contains 'question', 'image', 'diagnosis'
        """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    for item in data:
        question = item.get('question')# Question text
        image_path = item.get('image')# Image path
        answer = item.get('answer')# Correct answer / diagnosis result
        # Filter: ensure question and image exist, and image path matches expected format
        if question and image_path:
            results.append({'question': question, 'image': image_path, 'diagnosis': answer})

    return results# Return filtered question-image list

# Use LLM system prompt to generate description
def generate_description_with_openai(vl_api_key, api_key, question, image_path, system_prompt, k_shot_memory, model_name):
    # Call LLM to generate image description, combining system prompt and K-shot memory
    print(f"############################################################################New data task starts############################################################################")
    # Upload image
    filename = f"{datetime.now().timestamp()}_{os.path.basename(image_path)}"
    oss_url = aliyun_oss.oss_upload(image_path, filename)
    prompt01 = f"Image path: {oss_url}\n{system_prompt}"
    # Call LLM API to generate description
    desc = qwen_vl_max.call_with_messages(vl_api_key, prompt01, 0, 100, 0.1)
    print(desc)
    prompt02 = f"The image description is: {desc}, please help me extract the symptoms from the image description (the part after Symptom). The reference format example is:\n{k_shot_memory}. Please answer in English and cannot be None!"
    response = call_with_messages(api_key, prompt02, 0, 100, 0.1, model_name)
    return desc, response.replace("*","")


# System-level text encoder (initialize once)
TEXT_ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
# Use local model path
# TEXT_ENCODER = SentenceTransformer(
#     "/path/to/all-MiniLM-L6-v2",
#     device="cuda"  # Remove or change to "cpu" if GPU is not available
# )
def query_k_shot_memory(
    memory_file: str,
    query: str,
    k: int = 3
) -> Optional[List[str]]:
    """
    Based on similarity, extract Top-K descriptions from the description field in memory_file.
    Text encoder is built-in as a system module.
    """

    if not os.path.exists(memory_file):
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)
        return None

    try:
        with open(memory_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, list) or len(data) == 0:
        return None

    desc = []
    for item in data:
        if not isinstance(item, dict):
            continue
        d = item.get("description")
        if not d:
            continue
        if isinstance(d, str):
            desc.append(d)
        elif isinstance(d, list):
            desc.extend(d)

    if not desc or not isinstance(desc, list):
        return None

    # ===== Random subsampling (max 50 entries) =====
    if len(desc) > 50:
        desc = random.sample(desc, 50)

    # ===== Similarity calculation =====
    query_vec = TEXT_ENCODER.encode([query])
    desc_vecs = TEXT_ENCODER.encode(desc)
    sims = cosine_similarity(query_vec, desc_vecs)[0]
    top_k_idx = np.argsort(sims)[-k:][::-1]

    return [desc[i] for i in top_k_idx]