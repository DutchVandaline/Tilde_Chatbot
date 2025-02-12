import os
import torch
import torch.nn as nn
import torch.optim as optim
from Train_Step import train_step
from Test_Step import test_step

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from WorkStation.Train_Dataset import PreprocessedDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification

from WorkStation.model import TextViT
from LSTM import LSTMClassifier


def main():
    batch_size = 16
    num_epochs = 20
    learning_rate = 1e-5
    max_seq_len = 256
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 2
    num_classes = 9
    root_dir = "/home/junha/Tilde_Chatbot/Dataset_Tokenized"
    save_dir = "/home/junha/Tilde_Chatbot/models"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names_dir = "/home/junha/Tilde_Chatbot/Dataset_Tokenized/Train"

    kmbert_tokenizer = AutoTokenizer.from_pretrained("/home/idal/km-bert", do_lower_case=False)
    vocab_size = kmbert_tokenizer.vocab_size

    # Dataset and DataLoader
    train_dataset = PreprocessedDataset(os.path.join(root_dir, "Train"))
    test_dataset = PreprocessedDataset(os.path.join(root_dir, "Test"))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("DataLoader Process Completed")

    # model = LSTMClassifier(vocab_size=vocab_size,
    #                             embedding_dim=embedding_dim,
    #                             hidden_dim=hidden_dim,
    #                             output_dim=num_classes,
    #                             num_layers=num_layers,
    #                             dropout=0.5).to(device)
    model = AutoModelForSequenceClassification.from_pretrained(
        "/home/idal/km-bert", num_labels=num_classes
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(num_epochs)):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_accuracy = train_step(model, train_dataloader, loss_fn, optimizer, device)
        print(f"\nTrain Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        torch.cuda.empty_cache()

        # Validation step
        test_loss, test_accuracy, test_predictions, test_labels = test_step(model, test_dataloader, loss_fn,
                                                                            device, num_classes, epoch, class_names_dir)
        f1 = f1_score(test_labels, test_predictions, average='weighted')
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, F1-Score: {f1:.4f}")

        torch.cuda.empty_cache()

        epoch_model_path = os.path.join(save_dir, f"9_Class_Classification_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), epoch_model_path)
        print(f"Model for epoch {epoch + 1} saved to {epoch_model_path}")


if __name__ == "__main__":
    main()

