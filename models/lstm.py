import torch.nn as nn


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
