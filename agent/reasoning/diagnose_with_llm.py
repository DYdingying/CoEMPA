import pandas as pd
from openai import OpenAI

from agent.llm.gpt_4o_mini import call_with_messages


def diagnose_with_llm(dataset: str, symptom: str, api_key: str, model_name: str, crop, image_entry) -> str:
    """
    Use LLM to determine the disease based on the generated symptoms

    Parameters:
        dataset: CSV file path (must contain diagnosis column)
        symptom: Symptom description (str)
        api_key: OpenAI API key
    Returns:
        Disease name selected by LLM
    """
    # Read CSV
    df = pd.read_csv(dataset)
    if "diagnosis" not in df.columns:
        raise ValueError("CSV file must contain a diagnosis column")

    # Filter by crop
    df_filtered = df[df["crop"] == crop]

    # Get the corresponding diagnosis column
    diagnosis_list = df_filtered["diagnosis"].tolist()

    # Construct prompt
    prompt = f"""
        You are an agricultural disease diagnosis expert.
        The list of known possible diseases is as follows:
        {diagnosis_list}\n
        Based on the following symptom description, please select the most matching disease from the disease list, and only output the disease name without additional explanation.\n
        Crop name: {crop, image_entry}, Symptom description: {symptom}
    """
    # Call LLM
    response = call_with_messages(api_key, prompt, 0, 50, 0.1, model_name)

    return response