import pandas as pd
from torchtext.vocab import vocab
from collections import Counter, OrderedDict


def data_load():
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

    return train, valid, test_df


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
