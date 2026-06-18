# README

## Environment

Python Version:

```bash
Python >= 3.10
```

Install required packages:

```bash
pip install openai
pip install numpy
pip install pandas
pip install tqdm
pip install faiss-cpu
pip install scikit-learn
pip install pillow
pip install opencv-python
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Due to the large size of the full dataset, it is not included in the repository. Interested researchers may contact the authors to obtain access. Prepare the following files:

```text
dataset/
└── plantvillage-data/
    ├── disease_diagnosis.json
    ├── diseases-en.csv
    ├── image
    └── crop_classification.json
```

---

## Configuration

Before running the framework, please configure the following parameters in `main.py` or provide them through command-line arguments.

### API Keys

```python
--api_key        # LLM API key
--vl_api_key     # Vision-Language API key
```

### Dataset and File Paths

```python
--dataset                    # Path to diseases-en.csv
--json_path                  # Path to disease_diagnosis.json
--crop_classification_path   # Path to crop_classification.json
--memory_file                # Path to the memory repository
--output_file                # Output file for diagnostic results
```

Example:

```python
parser.add_argument(
    "--dataset",
    default="dataset/plantvillage-data/diseases-en.csv"
)

parser.add_argument(
    "--json_path",
    default="dataset/plantvillage-data/disease_diagnosis.json"
)

parser.add_argument(
    "--crop_classification_path",
    default="dataset/plantvillage-data/crop_classification.json"
)

parser.add_argument(
    "--memory_file",
    default="agent/plantvillage-data_gpt-3.5-turbo-1106.json"
)

parser.add_argument(
    "--output_file",
    default="plantvillage-data_process_results_gpt-3.5-turbo-1106.jsonl"
)
```

## Run

Execute:

```bash
python main.py
```

Or:

```bash
python main.py \
    --api_key YOUR_API_KEY \
    --vl_api_key YOUR_VL_API_KEY
```

---

