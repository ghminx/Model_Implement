# Inception v1(GoogleNet)

'''
모델의 깊이는 늘리면서 파라미터의 개수를 줄이기 위한 모델 
Inception 모듈을 기본 단위로 삼아 병렬로 1×1, 3×3, 5×5 컨볼루션과 풀링→1×1 컨볼루션을 수행
채널 축소용 1×1 컨볼루션을 도입해 큰 커널 연산의 연산량·파라미터 감소

'''
import torch 
import torch.nn as nn
import torchinfo 

class InceptionV1(nn.Module):
    def __init__(self, num_classes = 1000, aux_logits=True):
        super().__init__()
        
        self.aux_logits = aux_logits
        
        # basic layer 
        self.stem = nn.Sequential(                                       # size
            ConvBlock(3, 64, kernel_size = 7, stride = 2, padding = 3),  # 224 -> 112
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),            # 112 -> 56
            ConvBlock(64, 64, kernel_size = 1),                          # 56 -> 56
            ConvBlock(64, 192, kernel_size = 3, padding = 1),            # 56 -> 56
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)             # 56 -> 28
        )
        
        # ─── Inception 3a, 3b, pooling
        self.a3 = InceptionModule(192, 64, 96, 128, 16, 32, 32)          # out_dim -> 256
        self.b3 = InceptionModule(256, 128, 128, 192, 32, 96, 64)        # out_dim -> 480
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding = 1)  # shape : 28 -> 14
        
        # ─── Inception 4a–4e + pool 
        self.a4 = InceptionModule(480, 192,  96,208, 16, 48,  64)   # → 512
        self.b4 = InceptionModule(512, 160, 112,224, 24, 64,  64)   # → 512
        self.c4 = InceptionModule(512, 128, 128,256, 24, 64,  64)   # → 512
        self.d4 = InceptionModule(512, 112, 144,288, 32, 64,  64)   # → 528
        self.e4 = InceptionModule(528, 256, 160,320, 32,128, 128)   # → 832
        self.pool4 = nn.MaxPool2d(3, stride=2, padding=1)             # 14→7

        # ─── Inception 5a, 5b 
        self.a5 = InceptionModule(832, 256, 160,320, 32,128, 128)   # → 832
        self.b5 = InceptionModule(832, 384, 192,384, 48,128, 128)   # →1024

        # ─── Auxiliary Heads
        if aux_logits:
            self.aux1 = Auxiliary(512, num_classes)  # after a4
            self.aux2 = Auxiliary(528, num_classes)  # after d4
            
        else:
            self.aux1 = None
            self.aux2 = None
            
        # ─── Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(1024, num_classes)
        
        
    def forward(self, x):
        x = self.stem(x)
        
        x = self.a3(x)
        x = self.b3(x)
        x = self.pool3(x)
        
        x = self.a4(x)
        
        # a4 후 aux classifier(aux1)
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)
        
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)

        # d4 후 aux classifier(aux2)
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)
        
        x = self.e4(x)
        x = self.pool4(x)
        
        x = self.a5(x)
        x = self.b5(x)
        
        x = self.global_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        if self.aux_logits and self.training:  # 보조 Loss를 계산하기 위해서 학습시 aux1, aux2 return
            return x, aux1, aux2               # loss = L(main) + 0.3*L(aux1) + 0.3*L(aux2)
        
        else:
            return x

class InceptionModule(nn.Module):
    def __init__(self, in_channels, c1, c3_reduce, c3, c5_reduce, c5, pool_proj):
        super().__init__()
        
        # branch1 : 1x1 Conv
        self.branch1 = ConvBlock(in_channels, c1, kernel_size=1)
        
        # branch2 : 1x1 Conv -> 3x3 conv 
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, c3_reduce, kernel_size=1),
            ConvBlock(c3_reduce, c3, kernel_size=3, padding = 1)  # 입력 사이즈 유지를 위한 padding = 1
        )
        
        
        # branch3 : 1x1 Conv -> 5x5 Conv
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, c5_reduce, kernel_size=1),
            ConvBlock(c5_reduce, c5, kernel_size=5, padding=2)  # 입력 사이즈 유지를 위한 padding = 2 
        )        

        # branch4 : 3x3 maxpool -> 1x1 Conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, pool_proj, kernel_size=1)
        )        
        
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        return torch.cat([b1, b2, b3, b4], dim = 1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.block(x)
        return x         
    
# Inception의 중간 feature map에 보조적인 분류기를 달아 Vanishing gradient 완화 하는 역할
class Auxiliary(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        # (1) 5x5 AvgPooling 
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        
        # (2) 1x1 Conv + BN + ReLU
        self.block = ConvBlock(in_channels, 128, kernel_size = 1)
        
        # (3) FC Layer
        self.layers = nn.Sequential(
            nn.Linear(128*4*4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7)
        )
        
        # (4) Classifier
        self.classifier = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.avgpool(x)
        x = self.block(x)
        x = x.view(x.shape[0], -1)
        x = self.layers(x)
        x = self.classifier(x)
        
        return x


model = InceptionV1()
torchinfo.summary(model, input_size=(1, 3, 224, 224))
