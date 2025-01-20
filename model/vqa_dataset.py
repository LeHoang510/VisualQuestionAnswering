from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import ViTImageProcessor
from transformers import AutoTokenizer

from model.utils import *

class VQADatasetBasic(Dataset):
    def __init__(self, data, vocab, mapping, transform=None):
        self.transform = transform
        self.data = data
        self.max_seq_len = 20

        self.vocab = vocab
        self.classes, self.label2idx, self.idx2label = mapping

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = self.data[index]["image_path"]
        question = self.data[index]["question"]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        question = tokenize(question, self.max_seq_len, self.vocab)
        question = torch.tensor(question, dtype=torch.long)

        label = self.label2idx[self.data[index]["answer"]]
        label = torch.tensor(label, dtype=torch.long)

        return img, question, label

class VQADatasetAdvance(Dataset):
    def __init__(self, data, mapping, device, transform=None):
        self.transform = transform
        self.data = data
        self.device = device

        self.classes, self.label2idx, self.idx2label = mapping
        
        self.img_feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = self.data[index]["image_path"]
        question = self.data[index]["question"]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        img = self.img_feature_extractor(images=img, 
                                         return_tensors="pt")
        img = {k: v.to(self.device) for k, v in img.items()}

        question = self.tokenizer(question, 
                                  padding="max_length", 
                                  max_length=20, 
                                  truncation=True, 
                                  return_tensors="pt")
        question = {k: v.to(self.device).squeeze(0) for k, v in question.items()}

        label = self.label2idx[self.data[index]["answer"]]
        label = torch.tensor(label, dtype=torch.long).to(self.device)

        return img, question, label

class VQATransform():
    def get_transform(self, model, type):
        if model == "advance":
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.CenterCrop(180),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.RandomHorizontalFlip(),
                transforms.GaussianBlur(3),
            ])
        if model == "basic":
            if type == "train":
                return transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.CenterCrop(180),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                    transforms.RandomHorizontalFlip(),
                    transforms.GaussianBlur(3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            if type == "val":
                return transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])