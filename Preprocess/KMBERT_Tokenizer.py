import torch
from transformers import AutoTokenizer
import os
import json

# KM-BERT Tokenizer 로드
kmbert_tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-Medium", do_lower_case=False)

# Parameters
max_seq_len = 256
stride = 128
input_dir = "C:/junha/Datasets/ChatData_Processed/"
output_dir = "C:/junha/Datasets/ChatData_Tokenized/"

os.makedirs(output_dir, exist_ok=True)

def tokenize_with_sliding_window(text: str):
    """텍스트를 슬라이딩 윈도우 방식으로 토크나이즈하여 반환."""
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

        # 윈도우 길이가 max_seq_len보다 짧은 경우 패딩
        if len(input_ids_window) < max_seq_len:
            padding_length = max_seq_len - len(input_ids_window)
            input_ids_window = torch.cat([input_ids_window, torch.zeros(padding_length, dtype=torch.long)])
            attention_mask_window = torch.cat([attention_mask_window, torch.zeros(padding_length, dtype=torch.long)])

        tokenized_windows.append({'input_ids': input_ids_window, 'attention_mask': attention_mask_window})

    return tokenized_windows

# 하위 폴더 탐색 및 JSON 파일 처리
for root, _, files in os.walk(input_dir):
    for filename in files:
        if filename.endswith(".json"):
            filepath = os.path.join(root, filename)

            # JSON 파일 내용 읽기
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # "modifiedquery" 키가 존재할 경우에만 처리
            if "modifiedquery" in data:
                text = data["modifiedquery"]

                # 슬라이딩 윈도우 방식으로 토크나이즈
                tokenized_windows = tokenize_with_sliding_window(text)

                # 출력 경로 설정 (input_dir 구조를 유지)
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)

                # 각 윈도우를 고유한 이름으로 저장
                for i, window in enumerate(tokenized_windows):
                    output_path = os.path.join(output_subdir, f"{os.path.splitext(filename)[0]}_window_{i}.pt")
                    torch.save(window, output_path)

print("All files have been tokenized and saved with sliding windows.")
