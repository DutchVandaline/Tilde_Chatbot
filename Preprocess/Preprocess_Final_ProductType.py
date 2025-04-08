import pandas as pd
import json
import os

# 한글 클래스 이름과 영어 태그 매핑
class_name_mapping = {
    '생활화학제품': 'life',
    '의약품': 'medical',
    '농약': 'pesticide',
    '기타': 'etc',
}

# CSV 파일 읽기
train_file_path = 'C:/junha/Datasets/train_utf8.csv'
test_file_path = 'C:/junha/Datasets/test_utf8.csv'

df_train = pd.read_csv(train_file_path)
df_test = pd.read_csv(test_file_path)

# 필요한 컬럼 선택
columns_to_keep = ['질문수정', '노출제품유형']
df_train_filtered = df_train[columns_to_keep]
df_test_filtered = df_test[columns_to_keep]

# Train, Test 폴더 생성
output_base_dir = 'C:/junha/Datasets/FinalModified/'
train_dir = os.path.join(output_base_dir, 'Train')
test_dir = os.path.join(output_base_dir, 'Test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


# 함수: 데이터 저장
def save_data(df, target_dir):
    for index, row in df.iterrows():
        class_name = row['노출제품유형']
        english_class_name = class_name_mapping.get(class_name, 'etc')

        class_dir = os.path.join(target_dir, english_class_name)
        os.makedirs(class_dir, exist_ok=True)

        entry = {"modifiedquery": row['질문수정']}
        file_name = os.path.join(class_dir, f'row_{index}.json')

        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(entry, f, ensure_ascii=False, indent=4)


# 데이터 저장
save_data(df_train_filtered, train_dir)
save_data(df_test_filtered, test_dir)

print(f"Train, Test 폴더가 '{output_base_dir}'에 생성되었고, 각 폴더 아래에 클래스 폴더가 생성되었습니다.")
