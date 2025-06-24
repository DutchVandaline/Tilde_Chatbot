import os
import json
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

excel_path = 'C:/junha/Datasets/20250124_Chatdata.xlsx'
json_output_base = 'C:/junha/Datasets/5_Class_Classification_Dataset_json'
tokenized_output_base = 'C:/junha/Datasets/5_Class_Classification_Dataset_Tokenized'
tokenizer_path = '/home/idal/km-bert'
max_seq_len = 256
stride = 128

os.makedirs(json_output_base, exist_ok=True)
os.makedirs(tokenized_output_base, exist_ok=True)

class_name_mapping = {
    '경구': 'oral',
    '안구': 'eye',
    '피부': 'skin',
    '흡입': 'inhale',
    '기타': 'etc',
}

df = pd.read_excel(excel_path)
df_filtered = df[['질문수정', '노출유형']].sort_values(by='노출유형')

train_dir = os.path.join(json_output_base, 'Train')
test_dir = os.path.join(json_output_base, 'Test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for class_name in df_filtered['노출유형'].unique():
    mapped_name = class_name_mapping.get(class_name, str(class_name))
    cls_df = df_filtered[df_filtered['노출유형'] == class_name]
    train_df, test_df = train_test_split(cls_df, test_size=0.2, random_state=42)

    tdir = os.path.join(train_dir, mapped_name)
    vdir = os.path.join(test_dir, mapped_name)
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(vdir, exist_ok=True)

    for idx, row in train_df.iterrows():
        obj = {'modifiedquery': row['질문수정'], 'class': row['노출유형']}
        path = os.path.join(tdir, f'row_{idx}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=4)

    for idx, row in test_df.iterrows():
        obj = {'modifiedquery': row['질문수정'], 'class': row['노출유형']}
        path = os.path.join(vdir, f'row_{idx}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=4)

print(f"JSON split completed under: {json_output_base}")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, do_lower_case=False)

def tokenize_with_sliding_window(text: str):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=False)
    input_ids = tokens['input_ids'].squeeze(0)
    attention_mask = tokens['attention_mask'].squeeze(0)
    windows = []
    total_len = input_ids.size(0)
    num_windows = max(1, (total_len - max_seq_len + stride) // stride + 1)
    for i in range(num_windows):
        start = i * stride
        end = start + max_seq_len
        window_ids = input_ids[start:end]
        window_mask = attention_mask[start:end]
        if window_ids.size(0) < max_seq_len:
            pad_len = max_seq_len - window_ids.size(0)
            window_ids = torch.cat([window_ids, torch.zeros(pad_len, dtype=torch.long)])
            window_mask = torch.cat([window_mask, torch.zeros(pad_len, dtype=torch.long)])
        windows.append({'input_ids': window_ids, 'attention_mask': window_mask})
    return windows

for split in ['Train', 'Test']:
    in_base = os.path.join(json_output_base, split)
    out_base = os.path.join(tokenized_output_base, split)
    for root, _, files in os.walk(in_base):
        for file in files:
            if not file.endswith('.json'):
                continue
            filepath = os.path.join(root, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            text = data.get('modifiedquery', '')
            windows = tokenize_with_sliding_window(text)
            rel = os.path.relpath(root, json_output_base)
            out_dir = os.path.join(tokenized_output_base, rel)
            os.makedirs(out_dir, exist_ok=True)
            base_name = os.path.splitext(file)[0]
            for i, win in enumerate(windows):
                out_path = os.path.join(out_dir, f"{base_name}_window_{i}.pt")
                torch.save(win, out_path)

print(f"Tokenization completed under: {tokenized_output_base}")
