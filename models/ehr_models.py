import torch
import torch.nn as nn
import math

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.input_projection(x)  # 입력 데이터를 임베딩 차원으로 변환 (batch_size, seq_len, embed_dim)
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)  # 트랜스포머 인코더의 입력 형식에 맞게 차원 변환 (seq_len, batch_size, embed_dim)
        x = self.transformer_encoder(x)  # 트랜스포머 인코더를 통해 처리 (seq_len, batch_size, embed_dim)
        x = x.permute(1, 0, 2)  # 원래 차원 순서로 변환 (batch_size, seq_len, embed_dim)
        return x.mean(dim=1)  # 시퀀스 차원을 평균하여 임베딩 반환


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 위치 인코딩을 위한 행렬 생성
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 배치 차원을 추가하여 (1, max_len, embed_dim) 형태로 만듦
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 입력 데이터에 위치 인코딩을 더함
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

def get_time_series_embeddings(input_data, input_dim, embed_dim, num_heads, num_layers, dropout=0.1):
    model = TimeSeriesTransformer(input_dim, embed_dim, num_heads, num_layers, dropout)
    embeddings = model(input_data)
    return embeddings
