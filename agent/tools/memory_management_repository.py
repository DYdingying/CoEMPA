import csv
import os


import os
import json

def insert_entry_to_json(entry, json_file):
    """
    Insert question, code, description, and report from entry into the specified JSON file.

    Parameters:
        entry (dict): Dictionary containing data to be written, such as:
                      {
                          'question': 'xxx',
                          'code': 'xxx',
                          'description': 'xxx',
                          'report': 'xxx'
                      }
        json_file (str): Target JSON file path, default is 'data.json'
    """
    # If the file exists, load existing data; otherwise create an empty list
    if os.path.isfile(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Add the new entry to the list
    data.append({
        'id': entry.get('id', ''),
        'question': entry.get('question', ''),
        'code': entry.get('code', ''),
        'description': entry.get('description', ''),
        'report': entry.get('report', '')
    })

    # Write the data back to the JSON file (formatted output for readability)
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Successfully written to {json_file}")