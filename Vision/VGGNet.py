# VGGNet 구현 

'''
더 깊은(Deep) 네트워크가 성능을 높인다”는 가설을 검증하고자, 작은 3×3 필터를 연속으로 쌓아 깊이를 늘린 단순 구조
'''


import torch 
import torch.nn as nn 
from torchinfo import summary

cfg = {
    'VGG11' : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGGNet(nn.Module):
    def __init__(self, name, num_classes = 1000, init_weights = True, batch_norm = False):
        super().__init__()
        
        self.features = self.make_layers(cfg[name], batch_norm= batch_norm)
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        
        if init_weights:
            self._initialize_weights()


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x = x.view(-1, 7*7*512)
        x = self.classifier(x)
        return x
    
    
    def make_layers(self, cfg : list, batch_norm = False) -> nn.Sequential:
        in_channels = 3
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]    # layers = layers + [nn.MaxPool2d(v)]
            
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1)

                if batch_norm: 
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace = True)]
                else:  
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
                
        return nn.Sequential(*layers)
    
    
    def _initialize_weights(self):
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity='relu')
                if m.bias is not None:              # CNN에선 bias를 False로 하는경우도 있기 때문에 오류 방지를 위해 Not None
                    nn.init.constant_(m.bias, 0)    # bias 0으로 초기화
                    
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 1)    # fc layer는 항상 bias가 True이기 떄문에 Not None 사용안함
                nn.init.constant_(m.bias, 0)

model = VGGNet('VGG16', num_classes=10, batch_norm=True)
                    
summary(model, input_size=(1, 3, 224, 224))

