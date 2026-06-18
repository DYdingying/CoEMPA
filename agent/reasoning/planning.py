def generate_task_plan_from_keywords(crop, symptom, intent):
    """
    Generate a fixed task plan text based on a single list of keywords
    :param keywords: ['crop', 'symptom', 'disease']
    :return: Task plan string
    """

    task_plan = (
        f"1. Query the disease information related to {crop} {symptom} {intent} from the database;\n"
        "2. Invoke the LLM to automatically generate Python code for matching and retrieval;\n"
        "3. Execute the Python code in the code compiler;\n"
        "4. Compare the retrieval results and extracted image features with the correct entries in the database."
    )

    return task_plan