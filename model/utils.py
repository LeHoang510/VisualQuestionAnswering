import sys
import os.path as osp

BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)

import spacy
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import build_vocab_from_iterator

eng = spacy.load('en_core_web_sm')

def mapping_classes(dataset):
    classes = set([sample["answer"] for sample in dataset])
    label2idx = {label: idx for idx, label in enumerate(sorted(classes))}
    idx2label = {idx: label for idx, label in enumerate(sorted(classes))}
    return classes, label2idx, idx2label

def get_tokens(samples):
    # Split the question into tokens by using spacy tokenizer
    for sample in samples:
        question = sample["question"]
        yield [token.text for token in eng.tokenizer(question)]

def build_vocab(dataset):
    # Build the vocabulary from tokens that appear at least 2 times
    # and add special tokens (add them at the beginning)
    vocab = build_vocab_from_iterator(
        get_tokens(dataset),
        min_freq=2,
        specials=["pad", "unk", "eos", "sos"],
        special_first=True,
    )
    vocab.set_default_index(vocab["unk"])
    return vocab

def tokenize(question, max_seq_len, vocab):
    # Tokenize the question and map the tokens to their indices
    # output ex with padding: [10, 7, 15, 2, 20, 25, 30, 0, 0, 0, 0]
    tokens = [token.text for token in eng.tokenizer(question)]
    sequence = [vocab[token] for token in tokens]
    if len(sequence) < max_seq_len:
        sequence += [vocab["pad"]] * (max_seq_len - len(sequence))
    else:
        sequence = sequence[:max_seq_len]
    return sequence
