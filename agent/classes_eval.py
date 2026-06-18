import json
import re
from collections import defaultdict

PLANTS = [
    "Apple", "Bell Pepper", "Blueberry", "Cherry", "Corn", "Grape", "Orange",
    "Peach", "Potato", "Raspberry", "Soybean","Squash", "Strawberry", "Tomato"
]

def extract_plant(question: str):
    """Extract plant type from question text"""
    for plant in PLANTS:
        # Case-insensitive matching
        if re.search(rf"\b{plant}\b", question, re.IGNORECASE):
            return plant
    return "Unknown"


def evaluate_by_plant(json_path):
    # Read JSONL file
    with open(json_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    if not data:
        print("File is empty")
        return

    # Statistics per plant
    stats = defaultdict(lambda: {
        "total": 0,
        "code_correct": 0,
        "llm_correct": 0,
        "merged_correct": 0,
        "code_not_null": 0
    })

    for entry in data:
        question = entry.get("question", "")
        code_disease = entry.get("code_disease")
        llm_disease = entry.get("llm_disease")
        correct_disease = entry.get("correct_disease")

        plant = extract_plant(question)
        s = stats[plant]

        s["total"] += 1

        if code_disease == correct_disease:
            s["code_correct"] += 1

        if llm_disease == correct_disease:
            s["llm_correct"] += 1

        if code_disease == correct_disease or llm_disease == correct_disease:
            s["merged_correct"] += 1

        if code_disease is not None:
            s["code_not_null"] += 1

    # Output results for each plant
    for plant, s in stats.items():
        print(f"\n=== {plant} ===")
        print(f"Total samples: {s['total']}")
        print(f"Code accuracy: {(s['code_correct'] / s['total']) * 100:.2f}%")
        print(f"LLM accuracy: {(s['llm_correct'] / s['total']) * 100:.2f}%")
        print(f"Merged accuracy: {(s['merged_correct'] / s['total']) * 100:.2f}%")
        print(f"Completion rate: {(s['code_not_null'] / s['total']) * 100:.2f}%")


if __name__ == "__main__":
    json_file = "plantvillage-data_process_results_gpt-3.5-turbo.jsonl"
    evaluate_by_plant(json_file)