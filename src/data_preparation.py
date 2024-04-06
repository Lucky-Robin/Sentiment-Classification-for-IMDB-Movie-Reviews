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


def numericalize_and_pad(text, vocab, tokenizer, max_length):
    numericalized_text = [vocab[token] for token in tokenizer(text) if token in vocab]
    # 截断过长的序列
    numericalized_text = numericalized_text[:max_length]
    # 填充短序列
    padded_numericalized_text = numericalized_text + [vocab['<pad>']] * (max_length - len(numericalized_text))
    return padded_numericalized_text


def dataset_prepare(train_vocab, tokenizer, max_seq_length, train, valid, test_df):
    # 使用新的函数处理训练集和验证集中的文本
    train_numericalized_and_padded = [numericalize_and_pad(text, train_vocab, tokenizer, max_seq_length) for text in
                                      train['text']]
    valid_numericalized_and_padded = [numericalize_and_pad(text, train_vocab, tokenizer, max_seq_length) for text in
                                      valid['text']]
    test_numericalized_and_padded = [numericalize_and_pad(text, train_vocab, tokenizer, max_seq_length) for text in
                                     test_df['text']]
    # 直接将处理后的列表转换为Tensor
    train_data_padded = torch.tensor(train_numericalized_and_padded, dtype=torch.long)
    valid_data_padded = torch.tensor(valid_numericalized_and_padded, dtype=torch.long)
    test_data_padded = torch.tensor(test_numericalized_and_padded, dtype=torch.long)

    train_labels = torch.tensor(train['label'].values, dtype=torch.float)
    valid_labels = torch.tensor(valid['label'].values, dtype=torch.float)
    test_labels = torch.tensor(test_df['label'].values, dtype=torch.float)

    # 使用新的Tensor创建TensorDataset
    train_dataset = TensorDataset(train_data_padded, train_labels)
    valid_dataset = TensorDataset(valid_data_padded, valid_labels)
    test_dataset = TensorDataset(test_data_padded, test_labels)

    return train_dataset, valid_dataset, test_dataset
