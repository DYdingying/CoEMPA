import csv
import http
import random
import time
import json
import logging
# import Levenshtein
from typing import Dict, List, Optional, Union, Callable, Literal
import re
import pandas as pd
from termcolor import colored
from autogen.agentchat import UserProxyAgent
from agent.llm.gpt_4o_mini import call_with_messages
from agent.tools.calculate import normalized_levenshtein, ncd

# Initialize logging module
logger = logging.getLogger(__name__)

"""
    Crop disease intelligent diagnosis Agent
    Inherits from autogen.agentchat.UserProxyAgent,
    supports LLM-generated and auto-debugged Python code for disease matching diagnosis.
    """
class CropDiseaseAgent(UserProxyAgent):
    """
            Initialize the agent instance.

            Parameters:
                name: Agent name
                config_list: Configuration dictionary containing API config, model names, dataset path, etc.
                memory: Optional, memory information for context storage
            """
    def __init__(self, name: str, config_list: List[Dict], memory: Optional[List[Dict]] = None):
        # Call parent class initialization
        super().__init__(name=name, code_execution_config={"use_docker": False})
        # Read parameters from config
        self.config_list = config_list
        self.api_key = self.config_list[0]["api_key"]
        self.dataset = self.config_list[0]["dataset"]# CSV dataset path
        self.df = pd.read_csv(self.dataset)# Read CSV data
        self.model_name = self.config_list[0]["model_name"]# Main model
        self.code_model_name = self.config_list[0]["code_model_name"]# Code generation model
        self.memory = memory or []# Memory storage, default empty
        self.num_shots = 3# Few-shot example count
        self.question = '' # Current question
        self.generated_code = '' # LLM-generated code
        self.iteration_history = [] # Record auto-debug iteration history

    # Utility functions
    def extract_disease(self, diagnosis):
        # First extract the last folder name
        folder_name = diagnosis.split('/')[-2]
        # Split by comma, take the second part
        disease_name = folder_name.split(',')[1].strip()
        return disease_name

    def database_query(self, crop, symptom):

        """
        Query diagnosis in CSV based on crop and symptom
        """
        # Defense: if crop or symptom is empty, return None directly
        if not crop or not symptom:
            return None

        # Handle symptom as list or string
        if isinstance(symptom, list):
            symptom_lower = [s.lower() for s in symptom]
            symptom_filter = self.df["symptom"].str.lower().isin(symptom_lower)
        else:
            symptom_filter = self.df["symptom"].str.lower() == symptom.lower()
            # Convert crop field to lowercase for comparison
        crop_filter = self.df["crop"].str.lower() == crop.lower()

        # Filter matching rows
        result = self.df[crop_filter & symptom_filter]
        # Return diagnosis result or None
        if not result.empty:
            return result.iloc[0]["diagnosis"]
        else:
            return None


    def retrieve_k_shot(
        self,
        user_question: str,
        memory_file: str,
        top_k: int = 3,
        alpha: float = 0.5
    ) -> List[str]:
        try:
            with open(memory_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            print("OSError")
            return []

        records = data if isinstance(data, list) else []
        user_q = user_question.strip().lower()

        scored = []

        for rec in records:
            if not isinstance(rec, dict):
                continue

            question = rec.get("question")
            code = rec.get("code")

            if not question or not code:
                continue

            q = question.strip().lower()

            lev_d = normalized_levenshtein(user_q, q)
            ncd_d = ncd(user_q, q)

            final_d = alpha * lev_d + (1 - alpha) * ncd_d

            scored.append((final_d, code))

        if not scored:
            return []

        scored.sort(key=lambda x: x[0])

        return [code for _, code in scored[:top_k]]

    # LLM generates executable Python code
    def generate_executable_code(self, user_question, symptom, crop, disease_name, error_diagnosis, memory_file):
        """
        Use LLM to generate executable Python code for retrieving diagnosis results from CSV based on symptom similarity.
        """
        k_shot = self.retrieve_k_shot(user_question, memory_file)

        # Construct Prompt, instruct LLM to generate structured JSON output
        knowledge_prompt = f"""
        User question: {user_question}
        Historical cases: {k_shot}
        Crop: {crop}
        Symptom: {symptom}
        Database path for retrieval: {self.dataset}
        Please generate a reasonable similarity detection algorithm in Python code to retrieve the entry with the highest symptom similarity to: {symptom} for crop: {crop} from the database CSV file, and return the corresponding diagnosis (cannot be the following diseases: {error_diagnosis}). Do not import packages other than Python's built-in ones, e.g., sklearn.
        Strictly output in the following JSON structure:
        output = {{
        "crop": "...",
        "symptom": "...",
        "diagnosis": [...],
        }}
        Note: output is defined as a global variable. Provide only the code without any other content.
        """
        # print("=====================")
        # print(knowledge_prompt)# Call LLM to generate code
        code = call_with_messages(self.api_key, knowledge_prompt, 0, 500, 0.1, self.code_model_name)
        self.generated_code = code
        return code, knowledge_prompt

    def execute_code_with_debug(self, user_question, symptom, crop, disease_name, error_diagnosis, memory_file, max_attempts=5):
        """Execute LLM-generated code with auto-debugging, record iteration history"""
        global knowledge_prompt
        attempt = 0
        # Allow up to max_attempts auto-debug iterations
        while attempt < max_attempts:
            attempt += 1
            if attempt == 1:# First attempt: directly generate code
                code, knowledge_prompt = self.generate_executable_code(user_question, symptom, crop, disease_name, error_diagnosis, memory_file)
            else:# Subsequent iterations use corrected version
                code = self.generated_code
            print(f"========================Code execution failed, entering next iteration, current iteration count: {attempt}========================")
            # print(f"Generated code: " + self.generated_code)
            # Record current iteration information
            iteration_record = {"attempt": attempt, "code": code, "error": None, "result": None}

            try:
                # Set execution environment to prevent unsafe code execution
                local_vars = {}
                global_env = {
                    "__builtins__": __builtins__,
                    "csv": csv,
                    "json": json,
                    "re": re
                }
                print(code.strip().strip("`").replace("python", "", 1).strip())
                try:
                    exec(code.strip().strip("`").replace("python", "", 1).strip(),
                         global_env, global_env)
                except Exception as e:
                    print("=======")
                    print(e)
                    print("=======")

                # local_vars.update(global_env)
                # print("&"*100)
                # print("local_vars:", local_vars)
                # print("global_env:", global_env.keys())
                # print(global_env)

                # Check if output is defined in the LLM code
                if "output" in global_env:
                    iteration_record["result"] = global_env["output"]
                    self.iteration_history.append(iteration_record)
                    return iteration_record["result"], self.iteration_history, code
                # If output is not defined, treat as failure
                else:
                    raise ValueError("Code did not define 'result'")
                # Catch execution errors and generate new debug prompt
            except Exception as e:
                iteration_record["error"] = str(e)
                self.iteration_history.append(iteration_record)

                debug_prompt = f"""
                Original question: {knowledge_prompt}
                Crop symptom: {symptom}
                Code: {code}
                Error message: {str(e)}
                Please analyze the error cause, identify the error type and location, and provide the corrected executable code.
                Note: Provide only the corrected code without any other content.
                """
                debugged_code = call_with_messages(self.api_key, debug_prompt, 0, 500, 0.1, self.code_model_name)
                self.generated_code = debugged_code

        # Return error info if still fails after multiple debugging attempts
        return {"error": "Failed to execute code after multiple attempts."}, self.iteration_history, code

    def diagnose_crop_disease(self, user_question, planning, crop, symptom, diagnosis, memory_file):
        # Parse the absolutely correct disease name from the path
        disease_name = self.extract_disease(diagnosis)

        # Exact match query based on LLM-generated symptoms
        disease = self.database_query(crop, symptom)
        if disease == disease_name:
            return disease

        """AutoDebugLoop end-to-end auto-generate, execute, debug and return structured results with iteration history"""
        error_diagnosis = []
        for i in range(5):
            result, history, code = self.execute_code_with_debug(user_question, symptom, crop, disease_name, error_diagnosis, memory_file)

            if isinstance(result, str):
                result = json.loads(result)
            if result and "diagnosis" in result and result["diagnosis"]:
                c_disease = result['diagnosis'][0]  # Take the first item as the disease name predicted by the code
            else:
                c_disease = None
            if disease_name != c_disease:
                error_diagnosis.append(c_disease)
            else:
                break

        # print("#"*100)
        # print(result)
        return {
            "final_result": result,
            "iteration_history": history,
            "code": code
        }