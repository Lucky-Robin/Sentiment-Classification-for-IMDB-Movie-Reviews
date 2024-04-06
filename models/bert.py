import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 参数设置
SEED = 20
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 2
MAX_LEN = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 确保实验可重复
torch.manual_seed(SEED)

# 加载tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# 定义Dataset
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# 读取数据
df = pd.read_csv("./dataset/movie_train.csv")
df_train, df_val = train_test_split(df, test_size=0.25, random_state=SEED)
df_test = pd.read_csv("./dataset/movie_test.csv")
# 准备训练数据和测试数据
train_dataset = IMDBDataset(
    texts=df_train.text.to_numpy(),
    labels=df_train.label.to_numpy(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

val_dataset = IMDBDataset(
    texts=df_test.text.to_numpy(),
    labels=df_test.label.to_numpy(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

test_dataset = IMDBDataset(
    texts=df_test.text.to_numpy(),
    labels=df_test.label.to_numpy(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 计算数据集大小
train_size = len(train_loader.dataset)
val_size = len(val_loader.dataset)
test_size = len(test_loader.dataset)

# 输出训练集、验证集、测试集的大小
print(f"Training Set Size: {train_size}")
print(f"Validation Set Size: {val_size}")
print(f"Test Set Size: {test_size}")

# 模型
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model = model.to(device)

# 优化器和调度器
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


# 训练函数
def train_epoch(model, data_loader, optimizer, device, scheduler):
    model = model.train()
    losses = []
    correct_predictions = 0

    for data in tqdm(data_loader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)


# 评估函数
def eval_model(model, data_loader, device):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for data in tqdm(data_loader):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            preds = torch.argmax(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)


train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_loader,
        optimizer,
        device,
        scheduler
    )
    train_losses.append(train_loss)
    train_accuracies.append(train_acc.item())
    print()
    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
        model,
        val_loader,
        device
    )
    valid_losses.append(val_loss)
    valid_accuracies.append(val_acc.item())
    print()
    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()

# 绘制Loss图
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制Accuracy图
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(valid_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

def evaluate_test_set_with_roc(model, data_loader, device):
    model = model.eval()
    losses = []
    correct_predictions = 0

    actual_labels = []
    predicted_probs = []

    with torch.no_grad():
        for data in tqdm(data_loader):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            preds = torch.argmax(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            actual_labels.extend(labels.cpu().numpy())
            predicted_probs.extend(outputs.logits.softmax(dim=1)[:,1].cpu().numpy())

    accuracy = correct_predictions.double() / len(data_loader.dataset)
    mean_loss = np.mean(losses)

    # Calculate ROC AUC
    fpr, tpr, thresholds = roc_curve(actual_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for Test Set')
    plt.legend(loc="lower right")
    plt.show()

    return accuracy, mean_loss, roc_auc


test_acc, test_loss, test_auc = evaluate_test_set_with_roc(model, test_loader, device)
# 打印测试集的准确率
print(f'Test Loss: {test_loss:.3f} | Test Accuracy: {test_acc:.2f} | Test AUC: {test_auc:.2f}')
