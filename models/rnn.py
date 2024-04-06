import torch
import torch.nn as nn


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
