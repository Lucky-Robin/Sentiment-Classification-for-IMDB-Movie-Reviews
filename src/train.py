import pandas as pd
import torch
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from torchtext.vocab import vocab
from collections import Counter, OrderedDict
from src import data_preprocessing, data_preparation, tokenizer_build, train_utils
from models import rnn, lstm

# load data
train, valid, test_df = data_preprocessing.data_load()

# Build a vocabulary list
max_size = 25000
tokenizer = tokenizer_build.tokenizer_get('rnn')
train_vocab = tokenizer_build.build_vocab(train, tokenizer, max_size)
print(f"Total number of tokens in vocabulary: {len(train_vocab)}")

max_seq_length = 200

train_dataset, valid_dataset, test_dataset = data_preparation.dataset_prepare(train_vocab, tokenizer, max_seq_length,
                                                                              train, valid, test_df)

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM = len(train_vocab)  # 词汇表的大小
EMBEDDING_DIM = 100  # 选择的嵌入层维度
HIDDEN_DIM = 256  # RNN层的隐藏状态维度
OUTPUT_DIM = 1  # 输出层维度，对于二分类问题

model = rnn.VanillaRNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
# model = lstm.LSTMModel(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss().to(device)
model = model.to(device)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)


# Model training and evaluation
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        # 直接解包batch获取数据和标签
        text, labels = batch

        # 确保数据和模型都在同一设备上
        text = text.to(device)
        labels = labels.to(device)

        predictions = model(text).squeeze(1)

        loss = criterion(predictions, labels)
        acc = train_utils.binary_accuracy(predictions, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            # 直接解包batch获取数据和标签
            text, labels = batch

            # 确保数据和模型都在同一设备上
            text = text.to(device)
            labels = labels.to(device)

            predictions = model(text).squeeze(1)

            loss = criterion(predictions, labels)
            acc = train_utils.binary_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


N_EPOCHS = 40

best_valid_loss = float('inf')

train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_loader, criterion)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_acc)

    end_time = time.time()

    epoch_mins, epoch_secs = train_utils.epoch_time(start_time, end_time)
    scheduler.step(valid_loss)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "../checkpoints/rnn_model_saved")

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

# 绘制损失图
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制准确率图
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(valid_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend()
plt.show()

test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
