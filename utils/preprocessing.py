import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for _, text in data_iter:  
        yield tokenizer(text)

def build_vocab(train_data):
    vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=["<pad>", "<unk>"])
    vocab.set_default_index(vocab["<unk>"])  
    return vocab

def text_pipeline(vocab, text):
    return [vocab[token] for token in tokenizer(text)]

def label_pipeline(label):
    return 1 if label == "pos" else 0

def collate_batch(batch, vocab):
    label_list, text_list = [], []
    for label, text in batch:
        label_list.append(label_pipeline(label))  
        processed_text = text_pipeline(vocab, text)  
        text_list.append(torch.tensor(processed_text, dtype=torch.long))
    text_list = pad_sequence(text_list, batch_first=True, padding_value=vocab["<pad>"])
    label_list = torch.tensor(label_list, dtype=torch.long)  
    return text_list, label_list

def get_dataloaders(batch_size=32):
    train_iter, test_iter = IMDB(split=('train', 'test'))
    vocab = build_vocab(train_iter) 
    train_loader = DataLoader(list(train_iter), batch_size=batch_size, shuffle=True,
                              collate_fn=lambda batch: collate_batch(batch, vocab))
    test_loader = DataLoader(list(test_iter), batch_size=batch_size, shuffle=False,
                             collate_fn=lambda batch: collate_batch(batch, vocab))
    return train_loader, test_loader, vocab