import os
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

def test_step(model, dataloader, loss_fn, device, num_classes, epoch, class_names_dir):
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    total_val_samples = 0
    all_predictions = []
    all_labels = []
    all_probs = []

    class_names = sorted(os.listdir(class_names_dir))

    result_dir = f"/home/junha/Tilde_Chatbot/Result/Result_{epoch}"
    os.makedirs(result_dir, exist_ok=True)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # outputs이 튜플일 경우 대비
            loss = outputs.loss if hasattr(outputs, 'loss') else loss_fn(outputs.logits, labels)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            # Softmax 적용하여 확률값 계산
            probs = F.softmax(logits, dim=1)

            # Track validation loss
            val_running_loss += loss.item()

            # Calculate validation accuracy
            predictions = torch.argmax(logits, dim=1)
            val_correct += (predictions == labels).sum().item()
            total_val_samples += labels.size(0)

            # Store predictions, labels, and probabilities for metrics
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())  # 확률값 저장

    avg_val_loss = val_running_loss / len(dataloader)
    val_accuracy = val_correct / total_val_samples

    # Plot and save Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    plt.title(f"Confusion Matrix (Epoch {epoch})")
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(ticks=range(num_classes), labels=class_names, rotation=45)
    plt.yticks(ticks=range(num_classes), labels=class_names)
    plt.savefig(os.path.join(result_dir, f"confusion_matrix_epoch_{epoch}.png"))
    plt.close()

    # Plot and save ROC Curve for multi-class classification
    plt.figure(figsize=(10, 8))
    all_labels_onehot = torch.nn.functional.one_hot(torch.tensor(all_labels), num_classes=num_classes).numpy()
    all_probs_array = torch.tensor(all_probs).numpy()  # softmax 확률값 변환

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(all_labels_onehot[:, i], all_probs_array[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (Epoch {epoch})")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(result_dir, f"roc_curve_epoch_{epoch}.png"))
    plt.close()

    return avg_val_loss, val_accuracy, all_predictions, all_labels
