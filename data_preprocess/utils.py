import sys
import os.path as osp

BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)

import spacy
import torchtext
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

eng = spacy.load('en_core_web_sm')

def mapping_classes(dataset):
    classes = set([sample["answer"] for sample in dataset])
    label2idx = {label: idx for idx, label in enumerate(classes)}
    idx2label = {idx: label for label, idx in enumerate(classes)}
    return label2idx, idx2label

def get_tokens(samples):
    for sample in samples:
        question = sample["question"]
        yield [token.text for token in eng.tokenizer(question)]

def build_vocab(dataset):
    vocab = build_vocab_from_iterator(
        get_tokens(dataset),
        min_freq=2,
        specials=["pad", "unk", "eos", "sos"],
        special_first=True,
    )
    vocab.set_default_index(vocab["unk"])