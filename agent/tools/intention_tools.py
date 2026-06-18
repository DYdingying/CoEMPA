import re
import pandas as pd

def load_crop_list(csv_file: str, column: str = "crop") -> list:
    """Load crop name list from CSV file"""
    df = pd.read_csv(csv_file, encoding="utf-8")  # Read CSV file into DataFrame; change to "gbk" if encoding is incorrect
    crops = df[column].dropna().unique().tolist()# Take non-null unique values from specified column (default "crop") and convert to list
    # Convert to lowercase for matching
    return [c.strip().lower() for c in crops]# Remove leading/trailing spaces from each crop name, convert to lowercase, and return as list

# Initialize result dictionary containing Crop (identified crop), Symptom (symptom list), Intent (intent list)
def parse_description(text: str, crop_list: list) -> dict:
    result = {
        "Crop": None,
        "Symptom": [],
        "Intent": []
    }

    # -------- Crop identification --------
    # 1. Try to extract explicitly declared crop information from text using regex
    crop_match = re.search(r"(?:crop type is|crop is|the crop is|crop name)\s*[:\s]*([^\n.,]+)", text, re.IGNORECASE)
    if crop_match:
        # If regex matches, extract the first capture group, strip whitespace, and capitalize
        result["Crop"] = crop_match.group(1).strip().capitalize()
    else:
        # 2. If regex doesn't match, use the CSV crop list for text inclusion matching
        text_lower = text.lower()
        for crop in crop_list:
            if crop in text_lower:
                result["Crop"] = crop.capitalize()
                break  # Stop after finding the first match

    # -------- Symptom identification --------
    # Define a set of typical symptom expression patterns (regex) for matching common description fragments
    symptom_patterns = [
        r"leaf exhibits\s*([^.]+)",
        r"indications of\s*([^.]+)",
        r"include\s*([^.]+)",
        r"spots.*?([^.]+)"
    ]
    for pattern in symptom_patterns:
        # For each pattern, find all matches in the text (multiple symptom descriptions may appear in different places)
        for match in re.finditer(pattern, text, re.IGNORECASE):
            symptom_text = match.group(1).strip()
            if symptom_text not in result["Symptom"]:
                result["Symptom"].append(symptom_text)

    # -------- Intent identification --------
    intents = []# Temporary storage for identified intent labels
    # Add corresponding intent labels based on whether keywords related to disease diagnosis appear in the text
    if re.search(r"disease|symptom|diagnos|blotch|spot|infection|cause", text, re.IGNORECASE):
        intents.append("diagnose:disease_identification")# Disease identification intent
    if re.search(r"reason|cause|why", text, re.IGNORECASE):
        intents.append("diagnose:cause_analysis")# Cause analysis intent
    if re.search(r"treat|treatment|control|management|spray|recommend", text, re.IGNORECASE):
        intents.append("recommend:treatment")# Treatment/medication recommendation intent
    if re.search(r"fertiliz|pruning|watering|cultivation|management", text, re.IGNORECASE):
        intents.append("recommend:field_management")# Field management recommendation intent
    if re.search(r"monitor|spread|forecast|risk|tracking", text, re.IGNORECASE):
        intents.append("monitoring:disease_tracking")# Monitoring/risk tracking intent
    # If no intents match, provide default value "unspecified"
    result["Intent"] = intents or ["unspecified"]
    return result# Return dictionary containing Crop, Symptom, Intent

if __name__ == "__main__":
    # ---------- Configure CSV file path ----------
    csv_file = "D:\Agent\dataset\CDDM-data\diseases-en.csv" # Modify to your CSV file path
    crop_list = load_crop_list(csv_file, column="crop")

    # ---------- Test text ----------
    description = """
   "description": 
   1) Crop Name: Bell Pepper
   2) Symptom Description**: The image depicts a leaf from a bell pepper plant exhibiting symptoms associated with bacterial spot disease. The leaf shows several angular, water-soaked lesions that are predominantly dark brown to black in color. These spots are irregular in shape and vary in size, often surrounded by a yellow halo, indicating tissue damage. The lesions are distributed throughout the leaf, more concentrated towards the edges, and some spots appear to be merging, leading to larger areas of necrosis. The overall appearance of the leaf is wilting in the affected regions, suggesting that the plant may be under stress. There are no visible pest infestations observed in the image, focusing solely on the disease symptoms. The overall health of the leaf appears compromised due to the extent of the lesions present.
    """

    parsed = parse_description(description, crop_list)
    print(parsed)