import os
import torch
from torch.utils.data import Dataset


class PreprocessedDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        self.class_to_idx = self._get_class_to_idx(root_dir)
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.pt'):
                    class_name = os.path.basename(root) 
                    label = self.class_to_idx[class_name]

                    self.data.append({
                        'file_path': os.path.join(root, file),
                        'label': label
                    })

    def _get_class_to_idx(self, root_dir):
        class_names = sorted(os.listdir(root_dir)) 
        return {class_name: idx for idx, class_name in enumerate(class_names)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        data = torch.load(data_item['file_path'], weights_only=False)

        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        label = torch.tensor(data_item['label'], dtype=torch.long) 

        if torch.isnan(input_ids).any() or torch.isinf(input_ids).any():
            print(f"NaN or Inf detected in input_ids at index {idx}")
        if torch.isnan(attention_mask).any() or torch.isinf(attention_mask).any():
            print(f"NaN or Inf detected in attention_mask at index {idx}")


        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }
