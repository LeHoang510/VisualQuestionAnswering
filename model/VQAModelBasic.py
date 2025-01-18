from torchtext.vocab import Vocab
import torch.nn as nn
import torch
import timm

class VQAModelBasic(nn.Module):
    def __init__(
            self,
            n_classes: int,
            vocab: Vocab,
            img_model_name: str,
            embedding_dim: int,
            n_layers: int,
            hidden_dim: int,
            dropout: float=0.2,
    ):
        super(VQAModelBasic, self).__init__()
        self.img_encoder = timm.create_model(img_model_name, 
                                               pretrained=True,
                                               num_classes=hidden_dim)
        
        for param in self.img_encoder.parameters():
            param.requires_grad = True
        
        self.text_encoder = nn.Embedding(len(vocab), embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, img, text):
        img_features = self.img_encoder(img)
        
        text_features = self.text_encoder(text)
        lstm_out, _ = self.lstm(text_features)

        lstm_out = lstm_out[:, -1, :]
        combined = torch.cat((img_features, lstm_out), dim=1)
        x = self.fc1(combined)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


