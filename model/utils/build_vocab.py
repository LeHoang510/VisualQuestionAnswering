import spacy
from torchtext.vocab import build_vocab_from_iterator

eng = spacy.load('en_core_web_sm')

def get_tokens(samples):
    for sample in samples:
        question = sample["question"]
        yield [token.text for token in eng.tokenizer(question)]

def build_vocab(train_data):
    vocab = build_vocab_from_iterator(
        get_tokens(train_data),
        min_freq=2,
        specials=["pad", "unk", "eos", "sos"],
        special_first=True,
    )

    vocab.set_default_index(vocab["unk"])