import json
from sklearn.metrics import f1_score

def evaluate(json_path):
    # Read JSON file (one sample per line)
    with open(json_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    total = len(data)
    if total == 0:
        print("File is empty")
        return

    code_correct = 0
    llm_correct = 0
    merged_correct = 0
    code_not_null = 0

    y_true_code = []
    y_pred_code = []
    y_true_llm = []
    y_pred_llm = []
    y_true_merged = []
    y_pred_merged = []
    for entry in data:
        code_disease = entry.get("code_disease")
        llm_disease = entry.get("llm_disease")
        correct_disease = entry.get("correct_disease")

        # Count code accuracy
        if code_disease == correct_disease:
            code_correct += 1

        # Count llm accuracy
        if llm_disease == correct_disease:
            llm_correct += 1

        # Count merged accuracy
        if code_disease == correct_disease or llm_disease == correct_disease:
            merged_correct += 1

        # Count code completion rate
        if code_disease is not None:
            code_not_null += 1

        # code
        y_true_code.append(correct_disease)
        y_pred_code.append(code_disease if code_disease is not None else "__NONE__")

        # llm
        y_true_llm.append(correct_disease)
        y_pred_llm.append(llm_disease if llm_disease is not None else "__NONE__")

        # merged: correct if either is correct, otherwise wrong
        if code_disease == correct_disease or llm_disease == correct_disease:
            merged_pred = correct_disease
        else:
            merged_pred = "__WRONG__"  # Must be different from the correct label

        y_true_merged.append(correct_disease)
        y_pred_merged.append(merged_pred)

    f1_code = f1_score(y_true_code, y_pred_code, average="weighted")
    f1_llm = f1_score(y_true_llm, y_pred_llm, average="weighted")
    f1_merged = f1_score(y_true_merged, y_pred_merged, average="weighted")

    results = {
        "code_accuracy": code_correct / total,
        "llm_accuracy": llm_correct / total,
        "merged_accuracy": merged_correct / total,
        "completion_rate": code_not_null / total,
        "f1_code": f1_code,
        "f1_llm": f1_llm,
        "f1_merged": f1_merged

    }

    return results


if __name__ == "__main__":
    json_file = "CDDM-data_process_results_gpt-5_t=2.jsonl"  # Replace with your file path
    result = evaluate(json_file)
    print("Evaluation results:")
    for k, v in result.items():
        print(f"{k}: {v:.4f}")