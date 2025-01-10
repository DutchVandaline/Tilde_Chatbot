import os
import torch
from torch.utils.data import Dataset


class PreprocessedDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        self.class_to_idx = self._get_class_to_idx(root_dir)  # 클래스 → 정수 인덱스 매핑 생성

        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.pt'):
                    class_name = os.path.basename(root)  # 폴더 이름을 클래스 이름으로 사용
                    label = self.class_to_idx[class_name]  # 정수 인덱스로 변환

                    self.data.append({
                        'file_path': os.path.join(root, file),
                        'label': label
                    })

    def _get_class_to_idx(self, root_dir):
        """폴더 이름을 기준으로 클래스 → 정수 인덱스 매핑 생성."""
        class_names = sorted(os.listdir(root_dir))  # 정렬된 클래스 이름
        return {class_name: idx for idx, class_name in enumerate(class_names)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        data = torch.load(data_item['file_path'])

        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        label = torch.tensor(data_item['label'], dtype=torch.long)  # 정수형 label 반환

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }
