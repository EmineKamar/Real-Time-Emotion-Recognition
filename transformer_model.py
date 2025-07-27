import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerClassifier, self).__init__()

        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len=input_dim) → (batch_size, 1, input_dim)
        x = x.unsqueeze(1)  # [B, 1, input_dim]
        x = self.embedding(x)  # [B, 1, d_model]
        x = self.pos_encoder(x)  # Positional encoding
        x = self.transformer_encoder(x)  # [B, 1, d_model]
        x = x.mean(dim=1)  # Pooling: ortalama
        out = self.fc(x)   # [B, num_classes]
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pozisyon kodlaması hesapla
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # çift indeksler
        pe[:, 1::2] = torch.cos(position * div_term)  # tek indeksler
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
