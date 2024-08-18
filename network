import torch
import torch.nn as nn
import torch.nn.functional as F

from DGFEG.GCN import EGP
from dgf import DGF

from network.newbackbone import mctrans

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class Preblock(nn.Module):
    def __init__(self):
        super(Preblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
class DGFEG(nn.Module):
    def __init__(self, ):
        super(DGFEG, self).__init__()
        self.pre = Preblock()
        self.backbone = mctrans()
        self.conv_reduce_1 = BasicConv2d(64 * 2, 64, 3, 1, 1)
        self.conv_reduce_2 = BasicConv2d(128 * 2, 128, 3, 1, 1)
        self.conv_reduce_3 = BasicConv2d(256 * 2, 256, 3, 1, 1)
        self.conv_reduce_4 = BasicConv2d(512 * 2, 512, 3, 1, 1)

        self.decoder  = nn.Sequential(BasicConv2d(128, 64, 3, 1, 1), nn.Conv2d(64, 1, 3, 1, 1))
        self.decoder1 = nn.Sequential(BasicConv2d(256, 64, 3, 1, 1), nn.Conv2d(64, 1, 3, 1, 1))
        self.decoder2 = nn.Sequential(BasicConv2d(512, 64, 3, 1, 1), nn.Conv2d(64, 1, 3, 1, 1))
        self.decoder_final = nn.Sequential(BasicConv2d(64, 32, 3, 1, 1), nn.Conv2d(32, 2, 1))

        self.dgf = DGF(input_size=(64, 64), input_dim=64, hidden_dim=64)
        self.egp = EGP(2,1,64)

        self.upsample2x = nn.UpsamplingBilinear2d(scale_factor=2)

        self.decoder_module4 = BasicConv2d(768,128,3,1,1)
        self.decoder_module3 = BasicConv2d(256,64,3,1,1)
        self.decoder_module2 = BasicConv2d(128,64,3,1,1)

        self.decoder_module41 = BasicConv2d(768,128,3,1,1)
        self.decoder_module31 = BasicConv2d(256,64,3,1,1)
        self.decoder_module21 = BasicConv2d(128,64,3,1,1)

        self.conv2d = nn.Conv2d(2, 1, 3, 1, 1)
        self.convfinal = nn.Conv2d(2, 1, kernel_size=1)

    def forward(self, A, B):
        size = A.size()[2:]
        A = self.pre(A)
        B = self.pre(B)
        layer1_A,layer1_B,layer2_A,layer2_B, layer3_A,layer3_B,layer4_A,layer4_B = self.backbone(A,B)

        layer1 = torch.cat((layer1_B, layer1_A), dim=1)
        layer2 = torch.cat((layer2_B, layer2_A), dim=1)
        layer3 = torch.cat((layer3_B, layer3_A), dim=1)
        layer4 = torch.cat((layer4_B, layer4_A), dim=1)

        layer1 = self.conv_reduce_1(layer1)
        layer2 = self.conv_reduce_2(layer2)
        layer3 = self.conv_reduce_3(layer3)
        layer4 = self.conv_reduce_4(layer4)

        layer11 = torch.abs(layer1_B - layer1_A)
        layer22 = torch.abs(layer2_B - layer2_A)
        layer33 = torch.abs(layer3_B - layer3_A)
        layer44 = torch.abs(layer4_B - layer4_A)

        feature_map  = self.decoder2(layer4)
        feature_map1 = self.decoder1(layer3)
        feature_map2 = self.decoder (layer2)

        f43_ = self.decoder_module41(torch.cat([self.upsample2x(layer44), layer33], 1))
        f32_ = self.decoder_module31(torch.cat([self.upsample2x(f43_ ), layer22], 1))
        f21_ = self.decoder_module21(torch.cat([self.upsample2x(f32_ ), layer11], 1))

        f43 = self.decoder_module4(torch.cat([self.upsample2x(layer4),layer3],1))
        f32 = self.decoder_module3(torch.cat([self.upsample2x(f43),layer2],1))
        f21 = self.decoder_module2(torch.cat([self.upsample2x(f32), layer1], 1))

        fusion = self.DGF(f21_, f21)
        fusion_map = self.decoder_final(fusion)

        finalgcn = self.egp(fusion_map,A,B)
        finalgcn = self.convfinal(finalgcn)

        finalgcn = F.interpolate(finalgcn, size, mode='bilinear', align_corners=True)
        feature_map = F.interpolate(feature_map, size, mode='bilinear', align_corners=True)
        feature_map1 = F.interpolate(feature_map1, size, mode='bilinear', align_corners=True)
        feature_map2 = F.interpolate(feature_map2, size, mode='bilinear', align_corners=True)

        return feature_map,finalgcn, feature_map1, feature_map2
