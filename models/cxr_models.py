import torch
import torch.nn as nn
import math
from einops import rearrange
from torchvision import transforms
from PIL import Image

# 패치 임베딩 정의
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (batch_size, embed_dim, num_patches ** 0.5, num_patches ** 0.5)
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        return x

# 위치 인코딩 정의
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, num_patches):
        super(PositionalEncoding, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        return x + self.pos_embed

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_layers, num_heads, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = PositionalEncoding(embed_dim, num_patches)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)  # 패치 임베딩 수행
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # 클래스 토큰 추가
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_embed(x)  # 위치 인코딩 추가
        x = self.transformer_encoder(x)  # 트랜스포머 인코더 적용
        return x[:, 0, :]  # 클래스 토큰의 임베딩만 반환

def get_image_embedding(image_path, img_size, patch_size, in_channels, embed_dim, num_layers, num_heads, dropout=0.1):
    model = VisionTransformer(img_size, patch_size, in_channels, embed_dim, num_layers, num_heads, dropout)
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    input_image = transform(image).unsqueeze(0)
    embedding_vectors = model(input_image)
    return embedding_vectors
