import json
import pandas as pd

from transformers import AutoTokenizer
# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/mistral-7b-instruct-v0.2-bnb-4bit")
# File paths
CSV_FILENAME = "first_data.csv"
INSTRUCTIONS_FILENAME = "instructions.jsonl"
RESUME_FILENAME = "resumes.jsonl"
SKILLS_FILENAME = "skills.jsonl"
# Instruction for the model
INSTRUCTION = "Extract the skills from resume text"
# Read CSV file
df = pd.read_csv(CSV_FILENAME)
# Print table header
print(f"{'Total':<12}{'Instruction':<12}{'Resume':<12}{'Skills':<12}")
# Process each row in the CSV
for index, row in df.iterrows():
    story = row['Resume']
    summary = row['Skills string'].split(",")
    print(summary)
    print(type(summary))
    # Count tokens
    instruction_tokens = tokenizer(INSTRUCTION, return_tensors="pt",truncation=True)["input_ids"].shape[1]
    story_tokens = tokenizer(story, return_tensors="pt",truncation=True)["input_ids"].shape[1]
    summary_tokens = tokenizer(summary, return_tensors="pt",truncation=True)["input_ids"].shape[1]
    # Print table of tokens
    total_tokens = instruction_tokens + story_tokens + summary_tokens
    print(f"{total_tokens:<12}{instruction_tokens:<12}{story_tokens:<12}{summary_tokens:<12}")
    # Write to training files
    with open(INSTRUCTIONS_FILENAME, 'a') as file:
        file.write(json.dumps({"text": INSTRUCTION}) + "\n")
    with open(RESUME_FILENAME, 'a') as file:
        file.write(json.dumps({"text": story}) + "\n")
    with open(SKILLS_FILENAME, 'a') as file:
        file.write(json.dumps({"text": summary}) + "\n")