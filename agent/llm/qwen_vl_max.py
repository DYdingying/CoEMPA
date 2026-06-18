import os
import re
from http import HTTPStatus

import dashscope
def call_with_messages(vl_api_key, prompt, temperature, max_n_tokens, top_p):
    # Define function: send messages through dashscope's multimodal conversation API and return parsed text content
    # split_index = prompt.find(".png") + 4
    # url = prompt[:split_index]
    # prompt = prompt[split_index:]
    # Regex match image filenames in URL (supports png / jpg / jpeg / webp / gif)
    match = re.search(r'(https?://[^\n]+?\.(?:png|jpg|jpeg|webp|gif|JPG))', prompt, re.IGNORECASE)

    if match:
        url = match.group(1)  # Extract the first capture group as the image URL
        prompt = prompt.replace(url, '', 1)  # Remove URL, keep the remaining text as prompt
    else:
        url = None  # No image found
        # Construct message structure for multimodal API
    messages = [
        {
            "role": "user",
            "content": [
                {"image": url},# If url is None, this will contain {"image": None}, depending on how the API handles it
                {"text": prompt}# Remaining text prompt (the URL has been removed from the text above)
            ]
        }
    ]
    # Call dashscope API
    response = dashscope.MultiModalConversation.call(
        api_key=vl_api_key,
        model='qwen-vl-max',
        messages=messages,# Pass the constructed message list
        temperature=temperature,# Sampling temperature
        max_n_tokens=max_n_tokens,# Max generation tokens
        top_p=top_p,# Nucleus sampling parameter
        )
    # Check if response status code is successful (HTTP 200)
    if response.status_code == HTTPStatus.OK:
        # Assume response.output structure is similar to OpenAI style: read the content of the first choice message
        # Further assume content is a list and the first element contains the 'text' field
        content = response.output['choices'][0]['message']['content'][0]['text']
    else:
        content = "Sorry, I didn't understand" # If request fails, return default failure message
    return content

if __name__ == '__main__':
    vl_api_key = ""
    content = call_with_messages(vl_api_key,
        "Analyze this image",
        1, 100, 1)
    print(content)