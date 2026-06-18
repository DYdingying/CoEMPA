import argparse
import json
import re

import openai

from agent.reasoning.autodebugloop import CropDiseaseAgent
from agent.reasoning.diagnose_with_llm import diagnose_with_llm
from agent.reasoning.planning import generate_task_plan_from_keywords
from agent.perception.prompt_engineering import read_questions_and_images, query_k_shot_memory, \
    generate_description_with_openai
from agent.tools.intention_tools import parse_description
from agent.tools import knowledge_enhancement_tools
from agent.tools.memory_management_repository import insert_entry_to_json
from llm.gpt_4o_mini import call_with_messages
from tqdm import tqdm

"""  
    Main function: Implements the complete workflow of the agricultural agent from perception, reasoning to diagnosis and knowledge enhancement.
    Includes: image description, intent extraction, task planning, auto-debug execution and disease diagnosis evaluation.
"""
def main(args):
    # Initialize parameters and model configuration
    dataset = args.dataset
    api_key = args.api_key
    vl_api_key = args.vl_api_key
    model_name = args.model_name
    code_model_name = args.code_model_name
    is_google = args.is_google # Whether to use Google mode (used by knowledge_enhancement_tools)
    is_image_model = args.is_image_model
    json_path = args.json_path
    data_list = read_questions_and_images(json_path)# Each entry contains {"question": ..., "image": ...}
    # descriptions = [] # Store generated image descriptions
    output_file = args.output_file
    crop_classification_path = args.crop_classification_path
    memory_file = args.memory_file
    with open(crop_classification_path, "r", encoding="utf-8") as f:
        crops_list = json.load(f)
    print("Loaded data_list length:", len(data_list))


    ###### 1.Perception and Intent Extraction
    # 1.1 Prompt Engineering to get description
    # Define system prompt (for image description generation)
    SYSTEM_PROMPT = (
        f"""
        You are a crop pest and disease image description generator. When users upload images, provide detailed, accurate, and objective descriptions of plant leaves or related crops, including:
        1) Crop name: (Choose one from the following content: {crops_list});
        2) Symptom: for example (Leaf or plant condition, signs of pests and diseases, visible spots, bite marks, or abnormal parts and their location, color, shape, and distribution);
        NOTE: Use clear and orderly sentences to describe visible facts in detail. A fixed format must be used.
        """
    )

    # Initialize disease agent (for automatic diagnosis and debugging)
    crop_disease_agent = CropDiseaseAgent(
        name="crop_disease_agent",
        config_list=[{"api_key": api_key, "dataset": dataset, "model_name": model_name, "code_model_name": code_model_name}]
    )

    # Initialize accuracy counters
    accuracy_count_code, accuracy_count_llm, accuracy_count_cl, accuracy_rate_code, accuracy_rate_llm, accuracy_rate_cl = 0, 0, 0, 0, 0, 0
    complete_count_code, complete_rate_code = 0, 0
    total = len(data_list)# Total number of samples


    # Iterate through each data record
    # for idx, entry in enumerate(tqdm(data_list, total=len(data_list), desc="Processing", unit="item"), start=1):
    start_index = 0 # Can continue from this index if interrupted
    for idx, entry in enumerate(
            tqdm(data_list[start_index:], total=len(data_list) - start_index, desc="Processing", unit="item"),
            start=start_index + 1):

        # === Step 1: Image description generation ===
        # Call K-shot memory (example samples)
        k_shot_memory = query_k_shot_memory(memory_file, entry['question'])
        # Call description generation function: pass api_key, original question, image, system prompt and k-shot examples
        desc, symptom = generate_description_with_openai(vl_api_key, api_key, entry['question'], entry['image'], SYSTEM_PROMPT, k_shot_memory, model_name)
        print("\n"+"="*100)
        print(desc)
        # descriptions.append(desc)  # Append current description to descriptions list

        # 1.2 Call intent recognition tool to extract crop name, symptoms, and intent from the description and original question
        invocation_result = parse_description(desc,[])# parse_description returns a dictionary containing Crop/Symptom/Intent fields
        # crop, symptom, intent = invocation_result.get("Crop"), invocation_result.get("Symptom"), invocation_result.get("Intent")
        crop, intent = invocation_result.get("Crop"), invocation_result.get("Intent")
        # Clean crop/symptom: keep only English letters (adjust based on your data requirements)
        crop = re.sub(r'[^A-Za-z]', '', str(crop or ""))
        # symptom = re.sub(r'[^A-Za-z ]', '', str(symptom or ""))

        ###### 2.Automated Reasoning and Task Planning
        print("-"*100)
        print(f"Extracted disease symptom: {symptom}")
        # 2.1 Generate task plan based on crop name, symptoms, and intent
        planning = generate_task_plan_from_keywords(crop, symptom, intent)

        # 2.2 Action
        # 2.2.1 Use LLM to generate Python code for crop disease retrieval, call automatic diagnosis agent
        # 2.2.2 Execute the generated code in the compiler. If it runs without errors, output directly. If an error occurs,
        # pass the original question, code, and error information to Debugger for analysis, return the error reason to LLM
        # to correct the code and run again until it executes without errors.
        result = crop_disease_agent.diagnose_crop_disease(entry['question'], planning, crop, symptom, entry['image'], memory_file)
        final_result, code = result['final_result'], result['code']

        # If final_result is a string, parse it first to convert to dictionary
        if isinstance(final_result, str):
            final_result = json.loads(final_result)
            # Extract the disease name predicted by the code
        if final_result and "diagnosis" in final_result and final_result["diagnosis"]:
            c_disease = final_result['diagnosis'][0]# Take the first item as the disease name predicted by the code
        else:
            c_disease = None
        # c_disease = None

        # Use LLM to determine the disease based on the extracted symptoms
        l_disease = diagnose_with_llm(dataset, symptom, api_key, model_name, crop, entry['image']).strip()
        print(f"\n****************************Predicted disease: Code = {c_disease} LLM = {l_disease}****************************")

        # Get the correct disease name
        correct_disease = crop_disease_agent.extract_disease(entry['image'])

        # Save the processing result for this entry to record (for subsequent analysis)
        record = {
            "id": idx,
            "question": entry['question'],
            "description": desc,
            "crop": crop,
            "symptom": symptom,
            "intent": intent,
            "task_plan": planning,
            "code_disease": c_disease,
            "llm_disease": l_disease,
            "correct_disease": correct_disease
        }
        # Write in JSONL format to the output file, one JSON object per line (for easy subsequent analysis)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 2.2.3 Compare the key information from code execution with correct keywords in the database.
        # If code prediction or LLM prediction contains the correct answer, count it. If mismatch, output directly.
        if correct_disease in [c_disease, l_disease]:
            if c_disease == correct_disease:
                accuracy_count_code += 1
            if l_disease == correct_disease:
                accuracy_count_llm += 1
            accuracy_count_cl += 1
            # Count code successful predictions (i.e., code has output prediction result)
        if c_disease is not None:
            complete_count_code += 1

        # ###### 3.Knowledge-Augmented Diagnosis
        # # 3.1 Call knowledge enhancement tools, combine database query results and code output results,
        # introduce external agricultural knowledge base to improve knowledge response
        # 3.2 Call LLM to generate a complete report
        if correct_disease in [c_disease, l_disease]:
            # Call knowledge enhancement module to generate complete report: pass crop, disease name and various keys and flags
            # report = knowledge_enhancement_tools.report_generation(crop, correct_disease, api_key, is_google,"", vl_api_key, is_image_model, memory_file)
            report = ""
            # Construct entry to be stored in memory repository (contains question, code, description and report)
            memory_entry  = {
                'id': idx,
                'question': entry["question"],
                'code': str(code) if correct_disease == c_disease else "", # If code prediction is correct, store the code#todo
                'description': desc,# Pass all descriptions here (note: if each entry only needs current description, change to desc instead of descriptions)
                'report': report
            }
            # 3.3 Agent Memory Management Repository
            insert_entry_to_json(memory_entry, memory_file)

    # Calculate accuracy rates
    complete_rate_code = complete_count_code / total
    accuracy_rate_code, accuracy_rate_llm, accuracy_rate_cl = accuracy_count_code / total, accuracy_count_llm / total, accuracy_count_cl/total
    print(f"Code success rate: {accuracy_rate_code}")
    print(f"Model success rate: {accuracy_rate_llm}")
    print(f"Combined success rate: {accuracy_rate_cl}")
    print(f"Code completion rate: {complete_rate_code}")
    # Return summary dictionary for external calls to get results
    summary = {
        "total": total,
        "accuracy_rate_code": accuracy_rate_code,
        "accuracy_rate_llm": accuracy_rate_llm,
        "accuracy_rate_cl": accuracy_rate_cl,
        "complete_rate_code": complete_rate_code,
        "output_file": output_file
    }

    return summary

    # 3.3.1 Pass original question, code, image description, and report to evaluation model to analyze whether content is positive,
    # if yes, store it
    # process_all_reports()


if __name__ == "__main__":
    model = "gpt-3.5-turbo-1106"
    dataset = "plantvillage-data"
    parser = argparse.ArgumentParser(description="Agriculture Agent parameter configuration")
    parser.add_argument("--dataset", type=str,default=rf"D:\agriculture-Agent\dataset\{dataset}\diseases-en.csv")#Disease dataset path
    parser.add_argument("--api_key", type=str,default="")#OpenAI API
    parser.add_argument("--vl_api_key", type=str,default="")#Image recognition API key
    parser.add_argument("--model_name", type=str, default=model)#Main model name to use
    parser.add_argument("--code_model_name", type=str, default=model)#Main model name to use
    parser.add_argument("--is_google", action="store_true",default=True)#Whether to use Google mode (True enables it)
    parser.add_argument("--is_image_model", action="store_true",default=False)#Whether to use image analysis model (True enables it)
    parser.add_argument("--json_path", type=str,default=rf"D:\agriculture-Agent\dataset\{dataset}\disease_diagnosis.json")#Disease diagnosis JSON file path
    parser.add_argument("--output_file", type=str,default=f"{dataset}_process_results_{model}.jsonl")#Output file: write JSON objects line by line (one record per line)
    parser.add_argument("--crop_classification_path", type=str, default=rf"D:\agriculture-Agent\dataset\{dataset}\crop_classification.json")
    parser.add_argument("--memory_file", type=str, default=rf"D:\agriculture-Agent\agent\{dataset}_{model}.json")
    args = parser.parse_args()
    main(args)