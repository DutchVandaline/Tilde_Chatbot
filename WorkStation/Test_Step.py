import torch

def test_step(model, dataloader, loss_fn, device):
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    total_val_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)  # 1D 정수형 레이블

            # Forward pass
            outputs = model(input_ids, attention_mask)  # 출력 shape: [batch_size, num_classes]
            loss = loss_fn(outputs, labels)

            # Track validation loss
            val_running_loss += loss.item()

            # Calculate validation accuracy
            predictions = torch.argmax(outputs, dim=1)
            val_correct += (predictions == labels).sum().item()
            total_val_samples += labels.size(0)

    avg_val_loss = val_running_loss / len(dataloader)
    val_accuracy = val_correct / total_val_samples
    return avg_val_loss, val_accuracy
