import torch.nn as nn
import torch

from transformers import RobertaModel
from transformers import ViTModel

class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.model = RobertaModel.from_pretrained("roberta-base")
    
    def forward(self, x):
        x = self.model(**x)
        return x.pooler_output

class VisualEncoder(nn.Module):
    def __init__(self):
        super(VisualEncoder, self).__init__()
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224")
    
    def forward(self, x):
        x = self.model(**x)
        return x.pooler_output

class Classifier(nn.Module):
    def __init__(self,
                 n_classes: int,
                 hidden_dim: int=512,
                 dropout: float=0.2
    ):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(768 * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class VQAModelAdvance(nn.Module):
    def __init__(self, 
                 text_encoder: nn.Module=TextEncoder(),
                 visual_encoder: nn.Module=VisualEncoder(),
                 classifier: nn.Module=Classifier()
    ):
        super(VQAModelAdvance, self).__init__()
        self.text_encoder = text_encoder
        self.visual_encoder = visual_encoder
        self.classifier = classifier

    def forward(self, img, text):
        img_features = self.visual_encoder(img)
        text_features = self.text_encoder(text)

        x = torch.cat((img_features, text_features), dim=1)
        x = self.classifier(x)

        return x
    
    def freeze(self, visual=True, text=True, classifier=False):
        if visual:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
        if text:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        if classifier:
            for param in self.classifier.parameters():
                param.requires_grad = False
