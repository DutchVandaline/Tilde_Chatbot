import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

# 엑셀 파일 읽기
file_path = 'C:/junha/Datasets/filtered_excel_file.xlsx'
df = pd.read_excel(file_path)

# 필요한 컬럼 선택
columns_to_keep = ['질문수정', '노출제품유형']
df_filtered = df[columns_to_keep]

# 'class' 값 기준으로 정렬
df_filtered = df_filtered.sort_values(by='노출제품유형')

# Train, Test 폴더 생성
output_base_dir = 'C:/junha/Datasets/ChatData_Class/'
train_dir = os.path.join(output_base_dir, 'Train')
test_dir = os.path.join(output_base_dir, 'Test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 각 class별로 데이터를 분리하고 8:2 비율로 나누기
classes = df_filtered['노출제품유형'].unique()
for class_name in classes:
    class_data = df_filtered[df_filtered['노출제품유형'] == class_name]

    # Train-Test split (8:2 비율)
    train_data, test_data = train_test_split(class_data, test_size=0.2, random_state=42)

    # 각 class 폴더 생성
    train_class_dir = os.path.join(train_dir, str(class_name))
    test_class_dir = os.path.join(test_dir, str(class_name))
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # Train 데이터 저장
    for index, row in train_data.iterrows():
        entry = {
            "modifiedquery": row['질문수정'],
            "class": row['노출제품유형']
        }
        file_name = os.path.join(train_class_dir, f'row_{index}.json')
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(entry, f, ensure_ascii=False, indent=4)

    # Test 데이터 저장
    for index, row in test_data.iterrows():
        entry = {
            "modifiedquery": row['질문수정'],
            "class": row['노출제품유형']
        }
        file_name = os.path.join(test_class_dir, f'row_{index}.json')
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(entry, f, ensure_ascii=False, indent=4)

print(f"Train, Test 폴더가 '{output_base_dir}'에 생성되었고, 각 폴더 아래에 9개의 class 폴더가 생성되었습니다.")