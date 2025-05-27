import torch.nn as nn
import torchinfo


class DepthWise(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        
        self.dw = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.pw = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x
        
    
class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, int(32*alpha), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(32*alpha)),
            nn.ReLU(inplace=True),
        )
        
        self.dw1 = DepthWise(int(32*alpha), 64, 1)
        self.dw2 = DepthWise(64, 128, 2)
        self.dw3 = DepthWise(128, 128, 1)
        self.dw4 = DepthWise(128, 256, 2)
        self.dw5 = DepthWise(256, 256, 1)
        self.dw6 = DepthWise(256, 512, 2)
        
        layers = []
        for _ in range(5):
            layers.append(DepthWise(512, 512, 1))
        
        self.dw7 = nn.Sequential(*layers)
        self.dw8 = DepthWise(512, 1024, 2)
        self.dw9 = DepthWise(1024, 1024, 1)
        
        # classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)
        
                    
    def forward(self, x):
        x = self.stem(x)
        
        x = self.dw1(x)
        x = self.dw2(x)
        x = self.dw3(x)
        x = self.dw4(x)
        x = self.dw5(x)
        x = self.dw6(x)
        x = self.dw7(x)
        x = self.dw8(x)
        x = self.dw9(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        
        return x
    

torchinfo.summary(MobileNetV1(), input_size=(1, 3, 224, 224))
        
        
