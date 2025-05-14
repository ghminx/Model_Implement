# ResNet

'''
- **Degradation 문제**  
  네트워크를 깊게 쌓으면 일정 깊이 이상에서 학습·검증 성능이 오히려 떨어지는 현상이 나타남 (단순히 vanishing gradient 외에도 “degradation”이라고 불리는 현상)  

- **Skip Connection**  
  이 문제를 완화하기 위해 입력 \(x\)를 바로 다음 블록의 출력에 더해 주는 구조를 제시  

- **Identity Shortcut**  
    H(x) = F(x) + x
     
  여기서 F(x)는 블록 내부에서 학습되는 “잔차(Residual)”이며, 블록이 H(x) 전체를 학습하는 것이 아니라 F(x)=H(x)-x를 학습하도록 유도
   
- Bottleneck 구조 (ResNet-50 이상)* 
  1×1 Conv → 3×3 Conv → 1×1 Conv 순으로 채널을 줄였다가 늘리는 “병목형” 설계  
  - 중간 3×3 연산의 비용을 줄이면서도 표현력을 유지  
  
- Batch Normalization
  모든 Conv 레이어 뒤에 BN을 적용해 학습을 안정화하고 수렴 속도 향상  

'''
import torch 
import torch.nn as nn 

# ResNet-18/34
class BasicBlock(nn.Module):
    expansion = 1 # 출력 채널 수가 얼마나 확장되는지 (ResNet-18/34에서는 1)
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        
        self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding = 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )
        
        # 두번째 Conv 항상 stride = 1
        self.layer2 = nn.Sequential(
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding = 1, bias=False),
        nn.BatchNorm2d(out_channels),
        )
        
        self.residual = nn.Sequential()
        
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*self.expansion)
            )
            
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x 
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        x += self.residual(identity)
        x = self.relu(x)
        return x

# ResNet-50 이상
'''Input
  │
  ▼
1x1 Conv (축소, 채널 줄이기, 주로 out/4 채널)
  ↓
3x3 Conv (연산, 특징 추출)
  ↓
1x1 Conv (확장, 채널 복원)
  ↓
Shortcut 연결 (+ Input)
  ↓
ReLU
'''
class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # 3x3 Conv에만 stride가 변경됨
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )
        
        self.residual = nn.Sequential()
        
        if stride!=1 or in_channels != out_channels * self.expansion:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*self.expansion)
            )
            
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        

        identity = x
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x += self.residual(identity)
        x = self.relu(x)
        return x
      

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes = 1000):

        super().__init__()
        
        self.in_channels = 64
        
        # Stem Layer
        self.conv1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual Layers
        self.block1 = self._make_layer(block, out_channels=64, blocks=layers[0], stride=1)
        self.block2 = self._make_layer(block, out_channels=128, blocks=layers[1], stride=2)
        self.block3 = self._make_layer(block, out_channels=256, blocks=layers[2], stride=2)
        self.block4 = self._make_layer(block, out_channels=512, blocks=layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity='relu')
            
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    

    def _make_layer(self, block, out_channels, blocks, stride):

        strides = [stride] + [1] * (blocks - 1)
        
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)


    def forward(self, x):

        # Stem
        x = self.conv1(x)    
        
        # Residual Layers
        x = self.block1(x)   
        x = self.block2(x)   
        x = self.block3(x)   
        x = self.block4(x)   

        # Classifier
        x = self.avgpool(x)          
        x = torch.flatten(x, 1)      
        x = self.fc(x)               
        return x
      
class Model:
    def resnet18(self):
        return ResNet(BasicBlock, [2, 2, 2, 2])

    def resnet34(self):
        return ResNet(BasicBlock, [3, 4, 6, 3])

    def resnet50(self):
        return ResNet(BottleNeck, [3, 4, 6, 3])

    def resnet101(self):
        return ResNet(BottleNeck, [3, 4, 23, 3])

    def resnet152(self):
        return ResNet(BottleNeck, [3, 8, 36, 3])


if __name__ == "__main__":
    model = Model().resnet18()
    y = model(torch.randn(1, 3, 224, 224))
    print (y.size())