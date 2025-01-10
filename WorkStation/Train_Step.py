import torch

def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_running_loss = 0.0
    train_correct = 0
    total_train_samples = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)  # 정수형 레이블 (shape: [batch_size])

        # Forward pass
        outputs = model(input_ids, attention_mask)  # 출력 shape: [batch_size, num_classes]
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track training loss
        train_running_loss += loss.item()

        # Calculate training accuracy
        predictions = torch.argmax(outputs, dim=1)  # 가장 큰 값의 인덱스를 예측으로 선택
        train_correct += (predictions == labels).sum().item()
        total_train_samples += labels.size(0)

    avg_train_loss = train_running_loss / len(dataloader)
    train_accuracy = train_correct / total_train_samples
    return avg_train_loss, train_accuracy
