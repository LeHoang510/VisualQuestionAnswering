from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

from data_preprocess.utils import *

def VQADataset(Dataset):
    def __init__(self, data, max_seq_len=20, transform=None):
        self.transform = transform
        self.data = data
        self.max_seq_len = max_seq_len

        self.vocab = build_vocab(data)
        self.label2idx, self.idx2label = mapping_classes(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = self.data[index]["image_path"]
        question = self.data[index]["question"]

        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        
        question = tokenize(question, self.max_seq_len, self.vocab)
        question = torch.tensor(question, dtype=torch.long)

        label = self.label2idx[self.data[index]["answer"]]
        label = self.label2idx[label]

        return img, question, label