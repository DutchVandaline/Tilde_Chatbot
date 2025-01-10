import pandas as pd

# 엑셀 파일 읽기
file_path = 'C:/junha/Datasets/ChatData.xlsx'
df = pd.read_excel(file_path)

# 필요한 컬럼만 선택
columns_to_keep = ['질문수정', '노출제품유형']
df_filtered = df[columns_to_keep]

# 새로운 엑셀 파일로 저장
output_path = 'filtered_excel_file.xlsx'
df_filtered.to_excel(output_path, index=False)

print(f"필터링된 엑셀 파일이 '{output_path}'에 저장되었습니다.")
