import torch 
import torch.nn as nn 
from einops import rearrange
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, em_dim, dropout=0.1):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patchs = (img_size // patch_size) ** 2 
        
        self.proj = nn.Conv2d(in_channels, em_dim, kernel_size=patch_size, stride=patch_size)
        
        # CLS token 이미지의 제일 앞에 위치, 모든 패치로부터 정보를 모아 이미지 전체를 대표할 전역 토큰
        self.cls = nn.Parameter(torch.zeros(1, 1, em_dim))
        
        # Position Embedding : 학습 가능한 위치 정보 파라미터, cls토큰으로 인해 +1 
        self.pos = nn.Parameter(torch.zeros(1, 1 + self.num_patchs, em_dim))

        self.dropout = nn.Dropout(dropout)
        
        # initialize 
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        
        # x와 같은 차원으로 만들어진 cls 토큰을 배치 개수만큼만 확장 
        cls_token = self.cls.expand(B, -1, -1)   # (B 1 D)
        x = torch.cat([cls_token, x], dim = 1)
        
        x = x + self.pos
        x = self.dropout(x)            
        return x 
    
class MHA(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.n_heads = n_heads
        
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        
        self.scale = torch.tensor(d_model // n_heads) ** 0.5
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v):    
    
        Q = self.q(q)
        K = self.k(k)
        V = self.v(v)
        
        # s, h의 순서를 바꾸지 않으면 h에 대한 attention이 됨 (h d) @ (d h) -> (h h)  X  => (s d) @ (d s) => (s s)
        # 우리가 관계를 파악하고 싶은건 s  
        Q = rearrange(Q, 'b s (h d) -> b h s d', h = self.n_heads)
        K = rearrange(K, 'b s (h d) -> b h s d', h = self.n_heads)
        V = rearrange(V, 'b s (h d) -> b h s d', h = self.n_heads)
        
        attention_score = Q @ K.transpose(-2, -1) / self.scale
        
        attention_weights = torch.softmax(attention_score, dim=-1)
        
        attention = attention_weights @ V
        
        # concatenate num_heads * head_dim = d_model
        x = rearrange(attention, 'b h s d -> b s (h d)')
        
        x = self.o(x)
        x = self.dropout(x) 
        
        return x 
        
class MLP(nn.Module):
    def __init__(self, d_model, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        hidden_dim = int(d_model * mlp_ratio)
        self.fc = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )    
        
    def forward(self, x):
        x = self.fc(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, em_dim, n_heads, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(em_dim)
        self.mha = MHA(em_dim, n_heads, dropout)
        
        self.norm2 = nn.LayerNorm(em_dim)
        self.mlp = MLP(em_dim, mlp_ratio=4.0, dropout=dropout)
        
    def forward(self, x):

        x = x + self.mha(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        return x 
        

class ViT(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, em_dim, n_heads, depth, dropout=0.1, num_classes=1000):
        super().__init__()
        
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, em_dim, dropout)
        
        self.encoder = nn.ModuleList([
            EncoderLayer(em_dim, n_heads, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(em_dim)
        self.fc = nn.Linear(em_dim, num_classes)
        
    def forward(self, x):
        # x : (B C H W)
        x= self.patch_embedding(x)  # x : (B N+1, D)
        
        for layer in self.encoder:
            x = layer(x)
        
        x = self.norm(x)           # (B, N+1, D)
        cls_token = x[:, 0]        # (B, D)
        x = self.fc(cls_token)
        
        return x 
    
model = ViT(64, 4, 3, 512, 8, 6)