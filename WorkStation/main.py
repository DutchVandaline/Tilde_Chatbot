import os
import torch
import torch.nn as nn
import torch.optim as optim
from Train_Step import train_step
from Test_Step import test_step

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from WorkStation.Train_Dataset import PreprocessedDataset
from transformers import AutoTokenizer

from WorkStation.model import TextViT


def main():
    batch_size = 16
    num_epochs = 20
    learning_rate = 1e-4
    max_seq_len = 256
    num_classes = 9
    root_dir = "C:/junha/Datasets/ChatData_Tokenized"
    save_dir = "models/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kmbert_tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-Medium", do_lower_case=False)
    vocab_size = kmbert_tokenizer.vocab_size

    # Dataset and DataLoader
    train_dataset = PreprocessedDataset(os.path.join(root_dir, "Train"))
    test_dataset = PreprocessedDataset(os.path.join(root_dir, "Test"))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("DataLoader Process Completed")

    textViT = TextViT(vocab_size=vocab_size, max_seq_len=max_seq_len, num_classes=num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(textViT.parameters(), lr=learning_rate)

    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Train step
        train_loss, train_accuracy = train_step(textViT, train_dataloader, loss_fn, optimizer, device)
        print(f"\nTrain Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        torch.cuda.empty_cache()

        # Validation step
        test_loss, test_accuracy = test_step(textViT, test_dataloader, loss_fn, device)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        torch.cuda.empty_cache()

        epoch_model_path = os.path.join(save_dir, f"9_Class_Classification_epoch_{epoch + 1}.pth")
        torch.save(textViT.state_dict(), epoch_model_path)
        print(f"Model for epoch {epoch + 1} saved to {epoch_model_path}")

if __name__ == "__main__":
    main()
