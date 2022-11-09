import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from .unet_parts import *
from torchvision import models

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3,dilation=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3,dilation=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DownConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DownConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNET(nn.Module):

    def __init__(self, n_classes=8, n_channels =3, testing=False, bilinear=False):
        super(UNET, self).__init__()

        layer_dim = [64, 128, 256, 512 , 1024]

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.pool = nn.MaxPool2d(2, 2)      

        self.conv0_0 = DownConv(n_channels, layer_dim[0])
        self.conv1_0 = DownConv(layer_dim[0], layer_dim[1])
        self.conv2_0 = DownConv(layer_dim[1], layer_dim[2])
        self.conv3_0 = DownConv(layer_dim[2], layer_dim[3])
        factor = 2 if bilinear else 1
        self.conv4_0 = DownConv(layer_dim[3], layer_dim[4] // factor)

        self.conv4_1 = UpConv(layer_dim[4], layer_dim[3] // factor, bilinear)
        self.conv3_1 = UpConv(layer_dim[3], layer_dim[2] // factor, bilinear)
        self.conv2_1 = UpConv(layer_dim[2], layer_dim[1] // factor, bilinear)
        self.conv1_1 = UpConv(layer_dim[1], layer_dim[0], bilinear)
        self.conv0_1 = nn.Conv2d(layer_dim[0], n_classes, kernel_size=1)


    def forward(self, x):
        # print("x",x.size())  
        x0_0 = self.conv0_0(x)       
        # print("x0_0",x0_0.size())  
        x1_0 = self.conv1_0(self.pool(x0_0))
        # print("x1_0",x1_0.size())  
        x2_0 = self.conv2_0(self.pool(x1_0))
        # print("x2_0",x2_0.size())  
        x3_0 = self.conv3_0(self.pool(x2_0))
        # print("x3_0",x3_0.size())  
        x4_0 = self.conv4_0(self.pool(x3_0))
        # print("x4_0",x4_0.size())  

        x4_1 = self.conv4_1(x4_0, x3_0)
        # print("x4_1",x4_1.size()) 
        x3_1  = self.conv3_1(x4_1 , x2_0)
        # print("x3_1",x3_1.size()) 
        x2_1 = self.conv2_1(x3_1, x1_0)
        # print("x2_1",x2_1.size()) 
        x1_1 = self.conv1_1(x2_1, x0_0)
        # print("x1_1",x1_1.size()) 
        x0_1 = self.conv0_1(x1_1)      
        # print("x0_1",x0_1.size())        
        
        return x0_1  
        
class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, middle_channels=None):
        super().__init__()
        if not middle_channels:
            middle_channels = out_channels
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3,dilation=1, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3,dilation=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out     
        
class NestedUNet(nn.Module):
    def __init__(self, num_classes=8, input_channels=3, deep_supervision=True, **kwargs):
        super().__init__()

        layer_dim = [64, 128, 256, 512 , 1024]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, layer_dim[0])
        self.conv1_0 = VGGBlock(layer_dim[0], layer_dim[1])
        self.conv2_0 = VGGBlock(layer_dim[1], layer_dim[2])
        self.conv3_0 = VGGBlock(layer_dim[2], layer_dim[3])
        self.conv4_0 = VGGBlock(layer_dim[3], layer_dim[4])

        self.conv0_1 = VGGBlock(layer_dim[0]+layer_dim[1], layer_dim[0])
        self.conv1_1 = VGGBlock(layer_dim[1]+layer_dim[2], layer_dim[1])
        self.conv2_1 = VGGBlock(layer_dim[2]+layer_dim[3], layer_dim[2])
        self.conv3_1 = VGGBlock(layer_dim[3]+layer_dim[4], layer_dim[3])

        self.conv0_2 = VGGBlock(layer_dim[0]*2+layer_dim[1], layer_dim[0])
        self.conv1_2 = VGGBlock(layer_dim[1]*2+layer_dim[2], layer_dim[1])
        self.conv2_2 = VGGBlock(layer_dim[2]*2+layer_dim[3], layer_dim[2])

        self.conv0_3 = VGGBlock(layer_dim[0]*3+layer_dim[1], layer_dim[0])
        self.conv1_3 = VGGBlock(layer_dim[1]*3+layer_dim[2], layer_dim[1])

        self.conv0_4 = VGGBlock(layer_dim[0]*4+layer_dim[1], layer_dim[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(layer_dim[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(layer_dim[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(layer_dim[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(layer_dim[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(layer_dim[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


class UNetPlus(nn.Module):
    def __init__(self, num_classes=8, input_channels=3, deep_supervision=True, **kwargs):
        super().__init__()

        layer_dim = [64, 128, 256, 512 , 1024]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DownConv(input_channels, layer_dim[0])
        self.conv1_0 = DownConv(layer_dim[0], layer_dim[1])
        self.conv2_0 = DownConv(layer_dim[1], layer_dim[2])
        self.conv3_0 = DownConv(layer_dim[2], layer_dim[3])
        self.conv4_0 = DownConv(layer_dim[3], layer_dim[4])

        self.conv0_1 = DownConv(layer_dim[0]+layer_dim[1], layer_dim[0])
        self.conv1_1 = DownConv(layer_dim[1]+layer_dim[2], layer_dim[1])
        self.conv2_1 = DownConv(layer_dim[2]+layer_dim[3], layer_dim[2])
        self.conv3_1 = DownConv(layer_dim[3]+layer_dim[4], layer_dim[3])

        self.conv0_2 = DownConv(layer_dim[0]*2+layer_dim[1], layer_dim[0])
        self.conv1_2 = DownConv(layer_dim[1]*2+layer_dim[2], layer_dim[1])
        self.conv2_2 = DownConv(layer_dim[2]*2+layer_dim[3], layer_dim[2])

        self.conv0_3 = DownConv(layer_dim[0]*3+layer_dim[1], layer_dim[0])
        self.conv1_3 = DownConv(layer_dim[1]*3+layer_dim[2], layer_dim[1])

        self.conv0_4 = DownConv(layer_dim[0]*4+layer_dim[1], layer_dim[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(layer_dim[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(layer_dim[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(layer_dim[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(layer_dim[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(layer_dim[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output