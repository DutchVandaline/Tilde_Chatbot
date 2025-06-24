import torch
from transformers import AutoTokenizer
import os
import json

kmbert_tokenizer = AutoTokenizer.from_pretrained("/home/idal/km-bert", do_lower_case=False)

# Parameters
max_seq_len = 256
stride = 128
input_dir = "/home/junha/Tilde_Chatbot/Dataset"
output_dir = "/home/junha/Tilde_Chatbot/Dataset_Tokenized"

os.makedirs(output_dir, exist_ok=True)

def tokenize_with_sliding_window(text: str):
    tokens = kmbert_tokenizer(text, return_tensors="pt", padding='longest', truncation=True, max_length=512)
    input_ids = tokens['input_ids'].squeeze(0)
    attention_mask = tokens['attention_mask'].squeeze(0)

    tokenized_windows = []
    num_windows = (len(input_ids) - max_seq_len + stride) // stride + 1
    for i in range(num_windows):
        start = i * stride
        end = start + max_seq_len
        input_ids_window = input_ids[start:end]
        attention_mask_window = attention_mask[start:end]

        if len(input_ids_window) < max_seq_len:
            padding_length = max_seq_len - len(input_ids_window)
            input_ids_window = torch.cat([input_ids_window, torch.zeros(padding_length, dtype=torch.long)])
            attention_mask_window = torch.cat([attention_mask_window, torch.zeros(padding_length, dtype=torch.long)])

        tokenized_windows.append({'input_ids': input_ids_window, 'attention_mask': attention_mask_window})

    return tokenized_windows

for root, _, files in os.walk(input_dir):
    for filename in files:
        if filename.endswith(".json"):
            filepath = os.path.join(root, filename)

            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)

            if "modifiedquery" in data:
                text = data["modifiedquery"]

                tokenized_windows = tokenize_with_sliding_window(text)

                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)

                for i, window in enumerate(tokenized_windows):
                    output_path = os.path.join(output_subdir, f"{os.path.splitext(filename)[0]}_window_{i}.pt")
                    torch.save(window, output_path)

print("All files have been tokenized and saved with sliding windows.")
