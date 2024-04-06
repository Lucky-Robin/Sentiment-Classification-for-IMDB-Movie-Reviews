from torchtext.data.utils import get_tokenizer
from transformers import BertTokenizer
from src import data_preprocessing


def tokenizer_get(token_type):
    if token_type == 'rnn' or 'lstm':
        tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    elif token_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        print("Wrong tokenizer parameter, only 'rnn', 'lstm', 'bert' are allowed.")
        return 1
    return tokenizer


def build_vocab(train, tokenizer, max_size):
    train_vocab = data_preprocessing.build_vocab_from_dataframe(train, "text", tokenizer, max_size)
    print(f"Total number of tokens in vocabulary: {len(train_vocab)}")
    return train_vocab
