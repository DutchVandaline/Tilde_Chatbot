import os
import torch
import torch.nn as nn
from Test_Step import test_step

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from PreprocessedDataset import PreprocessedDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    batch_size = 16
    num_classes = 4
    root_dir = "/home/junha/Tilde_Chatbot/Dataset_Tokenized"
    class_names_dir = os.path.join(root_dir, "Train")
    model_checkpoint = "/home/junha/Tilde_Chatbot/Module/Models/4_Class_Classification_epoch_12.pth"

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("/home/idal/km-bert", do_lower_case=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        "/home/idal/km-bert", num_labels=num_classes
    )

    state_dict = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    test_dataset = PreprocessedDataset(os.path.join(root_dir, "Test"))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("DataLoader Process Completed")

    loss_fn = nn.CrossEntropyLoss()
    test_loss, test_accuracy, test_predictions, test_labels = test_step(
        model, test_loader, loss_fn, device, num_classes, 0, class_names_dir
    )
    f1 = f1_score(test_labels, test_predictions, average='weighted')
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, F1-Score: {f1:.4f}")

if __name__ == "__main__":
    main()
