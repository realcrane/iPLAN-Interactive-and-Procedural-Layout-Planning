from torch.nn import functional as F
import torchvision.models as models
from .basic import BasicModule
import torch.nn as nn
import numpy as np
import torch as t
import math

def get_upsampling_weight(input_channel, output_channel, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size,:kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((input_channel, output_channel, kernel_size, kernel_size))
    for i in range(input_channel):    
        weight[i,range(output_channel),:,:] = filt
    return t.from_numpy(weight).float()

def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            assert m.kernel_size[0] == m.kernel_size[1]
            initial_weight = get_upsampling_weight(
                m.in_channels, 
                m.out_channels, 
                m.kernel_size[0]
            )
            m.weight.data.copy_(initial_weight)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual

        return F.leaky_relu(out)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, dilation=dilation, padding=dilation, bias=False),
        self.bn2 = nn.BatchNorm2d(planes),
        self.conv3 = nn.Conv2d(planes, planes*4, 1, bias=False),
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.LeakyReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual

        return F.leaky_relu(out)

class ResNet(BasicModule):
    def __init__(self, name, block, layers, input_channel, output_channel, pretrained=False, pretrained_path = None):
        super(ResNet, self).__init__()
        self.name = name
        self.inplanes = 64
        self.conv1 = nn.Conv2d(input_channel, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dilation=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilation=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        initialize_weights(self)
        if pretrained:
            print('load the pretrained model...')
            if '34' in self.name :
                pretrained_model = '../pretrained_model/resnet34-333f7ec4.pth'
                resnet = models.resnet34()
            elif '18' in self.name:
                pretrained_model = '../pretrained_model/resnet18-5c106cde.pth'
                resnet = models.resnet18()
            if pretrained_path != None:
                pretrained_model = pretrained_path
            resnet.load_state_dict(t.load(pretrained_model))

            self.bn1 = resnet.bn1
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class FC(BasicModule):
    def __init__(self, name, input_channel, output_channel, reshape):
        super(FC, self).__init__()
        self.name = name
        self.reshape = reshape
        if input_channel < 2048:
            temp_channel = 512
        else:
            temp_channel = 2048

        fc_type = name.split('_')[-1]
        if fc_type == 'fc1':
            self.fc = nn.Sequential(
                nn.Conv2d(input_channel, temp_channel, 3, padding=1),
                nn.BatchNorm2d(temp_channel),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(temp_channel, output_channel, 1),
                nn.AdaptiveAvgPool2d(1)
            )
        if fc_type == 'fc2':
            self.fc = nn.Sequential(                
                nn.Linear(input_channel, temp_channel),
                nn.LeakyReLU(inplace=True),
                nn.Linear(temp_channel, temp_channel//2),             
                nn.LeakyReLU(inplace=True),
                nn.Linear(temp_channel//2, output_channel)
            )
        if fc_type == 'fc3':
            self.fc = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, 3, padding=1)                                          
            )  

        initialize_weights(self)
        
    def forward(self, x): 
        x = self.fc(x)          

        if self.reshape:
            x = x.view(x.size(0), -1)  

        return x

class UP(BasicModule):
    def __init__(self, name, input_channel, output_channel, reshape):
        super(UP, self).__init__()
        self.name = name
        self.reshape = reshape
        temp_channel = 1024
        input_channel = input_channel + 256

        up_type = name.split('_')[-1]
        if up_type == 'up1':
            dilations = [6, 12, 18, 24]
            self.aspp1 = nn.Sequential(
                nn.Conv2d(input_channel, temp_channel//2, 3, padding=dilations[0], dilation=dilations[0]),
                nn.BatchNorm2d(temp_channel//2),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(temp_channel//2, temp_channel//2, 1),
                nn.BatchNorm2d(temp_channel//2),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(temp_channel//2, temp_channel//2, 1)
            )
            self.aspp2 = nn.Sequential(
                nn.Conv2d(input_channel, temp_channel//2, 3, padding=dilations[1], dilation=dilations[1]),
                nn.BatchNorm2d(temp_channel//2),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(temp_channel//2, temp_channel//2, 1),
                nn.BatchNorm2d(temp_channel//2),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(temp_channel//2, temp_channel//2, 1)
            )
            self.aspp3 = nn.Sequential(
                nn.Conv2d(input_channel, temp_channel//2, 3, padding=dilations[2], dilation=dilations[2]),
                nn.BatchNorm2d(temp_channel//2),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(temp_channel//2, temp_channel//2, 1),
                nn.BatchNorm2d(temp_channel//2),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(temp_channel//2, temp_channel//2, 1)
            )
            self.aspp4 = nn.Sequential(
                nn.Conv2d(input_channel, temp_channel//2, 3, padding=dilations[3], dilation=dilations[3]),
                nn.BatchNorm2d(temp_channel//2),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(temp_channel//2, temp_channel//2, 1),
                nn.BatchNorm2d(temp_channel//2),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(temp_channel//2, temp_channel//2, 1)
            )     
            self.convT = nn.Sequential(
                nn.Conv2d(temp_channel//2, temp_channel//2, 3, padding=2, dilation=2),
                nn.ConvTranspose2d(temp_channel//2, output_channel, 8, stride=8)
            )                 
            
        initialize_weights(self)
            
    def forward(self, x):
        x1 = self.aspp1(x) 
        x2 = self.aspp2(x) 
        x3 = self.aspp3(x) 
        x4 = self.aspp4(x) 
        x = x1 + x2 + x3 + x4
        x = self.convT(x)
        
        if self.reshape:
            x = x.view(x.size(0), x.size(2), -1)

        return x
 
def model(module_name, model_name, **kwargs): 
    model_type = model_name.split('_')[0]
    name = module_name + "_" + model_type
    if "resnet18" in model_type:
        return ResNet(name, BasicBlock, [2, 2, 2, 2], **kwargs)
    if "resnet34" in model_type:
        return ResNet(name, BasicBlock, [3, 4, 6, 3], **kwargs)
    if "resnet50" in model_type:
        return ResNet(name, Bottleneck, [3, 4, 6, 3], **kwargs)
    if "resnet101" in model_type:
        return ResNet(name, Bottleneck, [3, 4, 23, 3], **kwargs)
    if "resnet152" in model_type:  
        return ResNet(name, Bottleneck, [3, 8, 36, 3], **kwargs)

def connect(module_name, model_name, **kwargs):
    model_type = model_name.split('_')[1]
    name = module_name + "_" + model_type
    if "fc" in model_type:
        return FC(name, **kwargs)
    if "up" in model_type:
        return UP(name, **kwargs)

def embedding(module_name, model_name, **kwargs):
    model_type = 'embed'
    name = module_name + "_" + model_type
    return Embedding(name, **kwargs)


class Embedding(BasicModule):
    def __init__(self, name, input_channel, output_channel, reshape= False):
        super(Embedding, self).__init__()
        self.name = name

        self.fc = nn.Sequential(
            nn.Linear(input_channel, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, output_channel, 3, 1, 1),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(inplace=True)
        )

        initialize_weights(self)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 4, 16, 16)
        x = self.conv(x)
        return x
