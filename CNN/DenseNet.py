import torch
import torch.nn as nn
import torchinfo
import torchvision
import torch.nn.functional as F

# torchvision.models.DenseNet()


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, 4*growth_rate, kernel_size=1, bias=False)
        
        self.norm2 = nn.BatchNorm2d(4*growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, x):
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        
        return torch.cat([x, out], dim=1)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm  = nn.BatchNorm2d(in_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv  = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool  = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x
    
class DenseNet(nn.Module):
    def __init__(self, growth_rate, num_blocks, num_classes=1000):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.denseblock = nn.ModuleList()
        self.transitions = nn.ModuleList()
        
        channels = 64
        for i, layers in enumerate(num_blocks):
            block= self._make_layers(DenseLayer, layers, channels, growth_rate)
            self.denseblock.append(block)
            
            # denseblock 채널수 업데이트
            channels += layers * growth_rate
            
            # Transition
            if i < len(num_blocks) - 1:  # 마지막 layer에는 추가 안함 
                in_ch = channels 
                out_ch = int(channels * 0.5)
                tran = Transition(in_ch, out_ch)
                self.transitions.append(tran)
                channels = out_ch
        
        # classifier
        self.norm_final = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        
        for i, block in enumerate(self.denseblock):
            x = block(x)
            
            if i < len(self.transitions):
                x = self.transitions[i](x)
            
        # classifier
        x = self.norm_final(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        
    # DenseBlock 생성 함수         
    def _make_layers(self, DenseLayer, num_blocks, in_channels, growth_rate):
        
        layers = []
        for _ in range(num_blocks):
            layers.append(DenseLayer(in_channels, growth_rate))
            in_channels += growth_rate 
            
        return nn.Sequential(*layers)
        

        
model = DenseNet(32, [6, 12, 24, 16])



