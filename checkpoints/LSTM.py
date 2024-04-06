import re
import pandas as pd
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab, build_vocab_from_iterator
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import time
import matplotlib.pyplot as plt

SEED = 30

# Set random seed for reproducibility
torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True

# Check for CUDA availability (GPU acceleration)
is_cuda = torch.cuda.is_available()
print("Cuda Status on system is {}".format(is_cuda))


# 数据清洗函数
def clean_text(text):
    text = text.lower()  # 转换为小写
    text = re.sub(r'[^\w\s]', '', text)  # 移除标点符号
    return text


# Load training and testing datasets using pandas' read_csv() function
train_df = pd.read_csv("../dataset/movie_train.csv")
test_df = pd.read_csv("../dataset/movie_test.csv")
print("Total training dataset shape:", train_df.shape)
print(" ")

# Shuffle the training dataset to improve generalization
train_df = train_df.sample(n=train_df.shape[0])

# Select text and label columns for data processing
train_df = train_df[["text", "label"]]
# print(train_df.head())
# Print the distribution of labels in the training dataset
print("Label distribution of total training dataset:", train_df.label.value_counts())
print(" ")
# Print the distribution of labels in the testing dataset
print("Label distribution of total testing dataset:", test_df.label.value_counts())
print(" ")

# Separate the training data into two classes based on the label
o_class = train_df.loc[train_df.label == 0, :]
l_class = train_df.loc[train_df.label == 1, :]

# Split the data into training and validation datasets
valid_o = o_class.iloc[:4000, :]
valid_l = l_class.iloc[:4000, :]
train_o = o_class.iloc[4000:, :]
train_l = l_class.iloc[4000:, :]

# Combine the two classes back into a single training dataset
train = pd.concat([train_o, train_l], axis=0)
# train = train.sample(n=train.shape[0])
# Print the shape of the final training dataset
print("Training dataset shape:", train.shape)
print(" ")
# Combine the two classes into a single validation dataset
valid = pd.concat([valid_o, valid_l], axis=0)
# valid = valid.sample(n=valid.shape[0])
# Print the shape of the validation dataset
print("Validation dataset shape:", valid.shape)
print(" ")
# Print the label distribution in the final training dataset
print("Label distribution of training dataset:", train.label.value_counts())
print(" ")
# Print the label distribution in the validation dataset
print("Label distribution of validation dataset:", valid.label.value_counts())
print(" ")

# train['text'] = train['text'].astype(str).apply(clean_text)
# valid['text'] = valid['text'].astype(str).apply(clean_text)

# Save processed datasets for future use
# train.to_csv("dataset/train.csv", index=False)
# valid.to_csv("dataset/valid.csv", index=False)

# Use SpaCy for tokenization
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')


def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)


from torchtext.vocab import vocab
from collections import Counter, OrderedDict


def build_vocab_from_dataframe(df, column_name, tokenizer, max_size):
    # 计算所有token的频率分布
    counter = Counter()
    for text in df[column_name]:
        counter.update(tokenizer(text))

    # 保留最常见的max_size-2个词，为<unk>和<pad>留出空间
    most_common_tokens = counter.most_common(max_size - 2)
    # 创建词汇表时，需要手动加入特殊符号
    final_vocab = vocab(OrderedDict(most_common_tokens), specials=['<unk>', '<pad>'])
    final_vocab.set_default_index(final_vocab['<unk>'])  # 设置默认的index为<unk>

    return final_vocab


# Build a vocabulary from text data for text numericalization
max_size = 25000
train_vocab = build_vocab_from_dataframe(train, "text", tokenizer, max_size)
print(f"Total number of tokens in vocabulary: {len(train_vocab)}")


def numericalize(text, vocab, tokenizer):
    return [vocab[token] for token in tokenizer(text) if token in vocab]


max_seq_length = 400


def numericalize_and_pad(text, vocab, tokenizer, max_length):
    numericalized_text = [vocab[token] for token in tokenizer(text) if token in vocab]
    # 截断过长的序列
    numericalized_text = numericalized_text[:max_length]
    # 填充短序列
    padded_numericalized_text = numericalized_text + [vocab['<pad>']] * (max_length - len(numericalized_text))
    return padded_numericalized_text


# 使用新的函数处理训练集和验证集中的文本
train_numericalized_and_padded = [numericalize_and_pad(text, train_vocab, tokenizer, max_seq_length) for text in
                                  train['text']]
valid_numericalized_and_padded = [numericalize_and_pad(text, train_vocab, tokenizer, max_seq_length) for text in
                                  valid['text']]

# 直接将处理后的列表转换为Tensor
train_data_padded = torch.tensor(train_numericalized_and_padded, dtype=torch.long)
valid_data_padded = torch.tensor(valid_numericalized_and_padded, dtype=torch.long)

train_labels = torch.tensor(train['label'].values, dtype=torch.float)
valid_labels = torch.tensor(valid['label'].values, dtype=torch.float)

# 使用新的Tensor创建TensorDataset
train_dataset = TensorDataset(train_data_padded, train_labels)
valid_dataset = TensorDataset(valid_data_padded, valid_labels)

# DataLoader部分保持不变
BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Define the Vanilla RNN model structure
class VanillaRNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(VanillaRNN, self).__init__()

        # 嵌入层
        self.embedding = nn.Embedding(input_dim, embedding_dim)

        # Vanilla RNN层
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)

        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text = [batch size, sent length]
        embedded = self.embedding(text)
        # embedded = [batch size, sent length, emb dim]

        output, hidden = self.rnn(embedded)
        # output = [batch size, sent length, hid dim]
        # hidden = [1, batch size, hid dim]

        assert torch.equal(output[:, -1, :], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0))


class LSTMModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)

        # 使用LSTM代替RNN
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)

        # LSTM的输出包括所有时刻的隐状态以及最后一个时刻的隐状态和细胞状态
        output, (hidden, cell) = self.lstm(embedded)

        # 使用最后一个时刻的隐状态
        hidden = hidden.squeeze(0)

        return self.fc(hidden)


# Initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM = len(train_vocab)  # 词汇表的大小
EMBEDDING_DIM = 100  # 选择的嵌入层维度
HIDDEN_DIM = 256  # RNN层的隐藏状态维度
OUTPUT_DIM = 1  # 输出层维度，对于二分类问题

# model = VanillaRNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
model = LSTMModel(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)


# optimizer = optim.SGD(model.parameters(), lr=1e-3)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss().to(device)
model = model.to(device)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)


# Calculating binary accuracy
def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


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
        acc = binary_accuracy(predictions, labels)

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
            acc = binary_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 300

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

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    scheduler.step(valid_loss)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "rnn_model_saved_78.77_64.78")

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
plt.legend()
plt.show()
