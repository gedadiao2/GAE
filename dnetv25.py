import warnings
from functools import partial
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
import math
from timm.models.layers import DropPath
from .modules.decoder_module import *
from Models import common
from Models import pvt_v2
from timm.models.vision_transformer import _cfg
from mmcv.cnn import ConvModule
from torch.nn import Conv2d, UpsamplingBilinear2d
from .mtblock1 import MTBlock
from thop import profile
import torch

class RB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)


class TB(nn.Module):
    def __init__(self):

        super().__init__()

        backbone = pvt_v2.PyramidVisionTransformerV2(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
        )

        checkpoint = torch.load("/home/ge/project/CNet/Models/pvt_v2_b3.pth")
        backbone.default_cfg = _cfg()
        backbone.load_state_dict(checkpoint)
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]

        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))

    def forward(self, x):
        pyramid = []
        B = x.shape[0]
        for i, module in enumerate(self.backbone):
            if i in [0, 3, 6, 9]:
                x, H, W = module(x)
            elif i in [1, 4, 7, 10]:
                for sub_module in module:
                    x = sub_module(x, H, W)
            else:
                x = module(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                pyramid.append(x)

        return pyramid

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

class SBA(nn.Module):

    def __init__(self, input_dim=64):
        super().__init__()

        self.input_dim = input_dim

        self.d_in1 = BasicConv2d(self.input_dim // 2, self.input_dim // 2, 1)
        self.d_in2 = BasicConv2d(self.input_dim // 2, self.input_dim // 2, 1)

        self.conv = BasicConv2d(self.input_dim, self.input_dim, 1)
        self.fc1 = nn.Conv2d(self.input_dim, self.input_dim // 2, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(self.input_dim, self.input_dim // 2, kernel_size=1, bias=False)

        self.Sigmoid = nn.Sigmoid()

    def forward(self, C_feature, T_feature):
        C_feature = self.fc1(C_feature)
        T_feature = self.fc2(T_feature)
        # print(C_feature.shape)
        # print(T_feature.shape)

        S_C_feature = self.Sigmoid(C_feature)
        S_T_feature = self.Sigmoid(T_feature)
        # print(S_C_feature)
        # print(S_C_feature.shape)
        # print(S_T_feature.shape)

        C_feature = self.d_in1(C_feature)
        T_feature = self.d_in2(T_feature)

        C_feature = C_feature + C_feature * S_C_feature + (1 - S_C_feature) * S_T_feature * T_feature

        T_feature = T_feature + T_feature * S_T_feature + (1 - S_T_feature) * S_C_feature * C_feature

        out = self.conv(torch.cat([T_feature, C_feature], dim=1))
        return out

class Double(nn.Module):
    def __init__(self,input_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = BasicConv2d(self.input_dim , self.input_dim , 1)
        self.fc2 = BasicConv2d(self.input_dim , self.input_dim , 1)
        self.conv1 = BasicConv2d(self.input_dim, self.input_dim, 1)
        self.conv2 = BasicConv2d(self.input_dim, self.input_dim, 1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self,x1,x2):
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        sx1 = self.Sigmoid(x1)
        sx2 = self.Sigmoid(x2)
        x1 = x1 + x1 * sx1 + sx1 * sx2 * x2
        x2 = x2 + x2 * sx2 + sx2 * sx1 * x1

        out1 = self.conv1(x1)
        out2 = self.conv2(x2)
        return out1,out2

class Encoder(nn.Module):
    def __init__(self,class_num=1):
        super(Encoder,self).__init__()
        self.pvt = TB()

        embed_dims = [64,128,320,512]
        self.mt0 = MTBlock(embed_dims[0], embed_dims[0])
        self.mt1 = MTBlock(embed_dims[1], embed_dims[1])
        self.mt2 = MTBlock(embed_dims[2], embed_dims[2])
        #self.mt3 = MTBlock(embed_dims[3], embed_dims[3]//4)

    def forward(self, x):
        pvt = self.pvt(x)
        t0 = pvt[0]
        #print(t0.shape)
        t1 = pvt[1]
        #print(t1.shape)
        t2 = pvt[2]
        #print(t2.shape)
        t3 = pvt[3]
        #print(t3.shape)

        d0 = self.mt0(t0)
        d1 = self.mt1(t1)
        d2 = self.mt2(t2)
        #print(d0.shape)

        return d0,d1,d2,t3

class upsample_2x(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=4, stride=2, padding=1, dilation=1):
        super(upsample_2x, self).__init__()
        self.conv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class F(nn.Module):
    def __init__(self,c1_in_channels = 64,c2_in_channels = 128,c3_in_channels = 320,c4_in_channels = 512):
        super(F, self).__init__()

        self.fuse3 = ConvModule(in_channels=c3_in_channels * 2, out_channels=c3_in_channels, kernel_size=1,
                                norm_cfg=dict(type='BN', requires_grad=True))
        self.fuse2 = ConvModule(in_channels=c2_in_channels * 2, out_channels=c2_in_channels, kernel_size=1,
                                norm_cfg=dict(type='BN', requires_grad=True))
        self.fuse1 = ConvModule(in_channels=c1_in_channels * 2, out_channels=c1_in_channels, kernel_size=1,
                                norm_cfg=dict(type='BN', requires_grad=True))
        self.up4 = upsample_2x(c4_in_channels, c3_in_channels)
        self.up3 = upsample_2x(c3_in_channels, c2_in_channels)
        self.up2 = upsample_2x(c2_in_channels, c1_in_channels)

    def forward(self,x1,x2,x3,x4):
        z4 = self.up4(x4)
        z3 = self.fuse3(torch.cat([x3,z4],dim=1))
        z3 = self.up3(z3)
        z2 = self.fuse2(torch.cat([x2,z3],dim=1))
        z2 = self.up2(z2)
        z1 = self.fuse1(torch.cat([x1,z2],dim=1))

        return z1

class downsample_2x(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=4, stride=2, padding=1, dilation=1):
        super(downsample_2x, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class Stem(nn.Module):
    def __init__(self,in_c = 3, min_c = 32, out_c = 64,embed_dims=[64,128,256,512]):
        super(Stem,self).__init__()
        self.blocks = nn.Sequential(
        nn.Conv2d(in_c, min_c, kernel_size=3,stride=2,padding=1),
        RB(min_c, out_c),
        nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
        RB(out_c, out_c))

        self.DWT2 = common.DWT()
        self.DWT3 = common.DWT()
        self.DWT4 = common.DWT()

        self.IWT2 = common.IWT()
        self.IWT3 = common.IWT()
        self.IWT4 = common.IWT()

        self.rb1 = nn.Sequential(RB(4*embed_dims[0], embed_dims[1]),RB(embed_dims[1], embed_dims[1]))
        self.rb2 = nn.Sequential(RB(4*embed_dims[1], embed_dims[2]),RB(embed_dims[2], embed_dims[2]))
        self.rb3 = nn.Sequential(RB(4*embed_dims[2], embed_dims[3]),RB(embed_dims[3], embed_dims[3]))

        self.r1 = nn.Sequential(RB(2*embed_dims[0], embed_dims[0]), RB(embed_dims[0], embed_dims[0]))
        self.r2 = nn.Sequential(RB(2*embed_dims[1], embed_dims[1]), RB(embed_dims[1], embed_dims[1]))
        self.r3 = nn.Sequential(RB(2*embed_dims[2], 320), RB(320, 320))


    def forward(self,x):
        outs = []
        for block in self.blocks:
            x = block(x)
        x1 = x

        x = self.DWT2(x1) #44,256
        outs.append(x)
        x2 = self.rb1(x) #44,128

        x = self.DWT3(x2) #22,512
        outs.append(x)
        x3 = self.rb2(x) #22,256

        x = self.DWT4(x3) #11,1024
        outs.append(x)
        x4 = self.rb3(x) #11,512

        #print(x.shape)
        x = self.IWT4(outs[2]) #22,256
        x3 = self.r3(torch.cat((x3,x),dim=1)) #22,320
        #print(x3.shape)

        x = self.IWT3(outs[1]) #44,128
        x2 = self.r2(torch.cat((x2,x),dim=1)) #44,128

        x = self.IWT2(outs[0]) #88,64
        x1 = self.r1(torch.cat((x1,x),dim=1)) #88,64

        return x1,x2,x3,x4


class TNet(nn.Module):
    def __init__(self,class_num=1):
        super().__init__()
        self.class_num = class_num
        self.e = Encoder()
        self.s = Stem()
        self.f = F()
        self.fn = F()
        self.dropout = nn.Dropout(0.1)
        self.dropoutn = nn.Dropout(0.1)
        self.linear_pred = Conv2d(64, self.class_num, kernel_size=1)
        self.linear_predn = Conv2d(64, self.class_num, kernel_size=1)
        self.SBA0 = SBA(input_dim=64)
        self.SBA1 = SBA(input_dim=128)
        self.SBA2 = SBA(input_dim=320)
        self.SBA3 = SBA(input_dim=512)
        self.y1 = ConvModule(in_channels=64, out_channels=1, kernel_size=1,
                                norm_cfg=dict(type='BN', requires_grad=True))
        self.y2 = ConvModule(in_channels=128, out_channels=1, kernel_size=1,
                             norm_cfg=dict(type='BN', requires_grad=True))
        self.y3 = ConvModule(in_channels=320, out_channels=1, kernel_size=1,
                             norm_cfg=dict(type='BN', requires_grad=True))
        self.y4 = ConvModule(in_channels=512, out_channels=1, kernel_size=1,
                             norm_cfg=dict(type='BN', requires_grad=True))
        self.y5 = ConvModule(in_channels=64, out_channels=1, kernel_size=1,
                             norm_cfg=dict(type='BN', requires_grad=True))
        self.y6 = ConvModule(in_channels=128, out_channels=1, kernel_size=1,
                             norm_cfg=dict(type='BN', requires_grad=True))
        self.y7 = ConvModule(in_channels=320, out_channels=1, kernel_size=1,
                             norm_cfg=dict(type='BN', requires_grad=True))
        self.y8 = ConvModule(in_channels=512, out_channels=1, kernel_size=1,
                             norm_cfg=dict(type='BN', requires_grad=True))

    def forward(self,x):
        en1,en2,en3,en4 = self.e(x)
        s1,s2,s3,s4 = self.s(x)

        e1 = self.SBA0(en1, s1)
        e2 = self.SBA1(en2, s2)
        e3 = self.SBA2(en3, s3)
        e4 = self.SBA3(en4, s4)

        fres = self.f(e1, e2, e3, e4)
        p = self.dropout(fres)
        p = self.linear_pred(p)
        up = UpsamplingBilinear2d(scale_factor=4)
        features = up(p)
        up1 = UpsamplingBilinear2d(scale_factor=4)
        up2 = UpsamplingBilinear2d(scale_factor=8)
        up3 = UpsamplingBilinear2d(scale_factor=16)
        up4 = UpsamplingBilinear2d(scale_factor=32)
        e1 = up1(e1)
        e1 = self.y1(e1)
        e2 = up2(e2)
        e2 = self.y2(e2)
        e3 = up3(e3)
        e3 = self.y3(e3)
        e4 = up4(e4)
        e4 = self.y4(e4)

        # fresn = self.fn(s1, s2, s3, s4)
        # pn = self.dropoutn(fresn)
        # pn = self.linear_predn(pn)
        # up = UpsamplingBilinear2d(scale_factor=4)
        # featuresn = up(pn)
        # s1 = up1(s1)
        # s1 = self.y5(s1)
        # s2 = up2(s2)
        # s2 = self.y6(s2)
        # s3 = up3(s3)
        # s3 = self.y7(s3)
        # s4 = up4(s4)
        # s4 = self.y8(s4)

        return features
        #return e1,e2,e3,e4,features,s1,s2,s3,s4,en1,en2,en3,en4

if __name__=='__main__':
    img = torch.randn(1, 3, 352, 352).cuda()
    t = TNet().cuda()
    macs, params = profile(t, inputs=(img,))
    #e1,e2,e3,e4,feature = t(img)
    feature= t(img)
    # print(e1.shape)
    # print(e2.shape)
    # print(e3.shape)
    # print(e4.shape)
    print(feature[4].shape)
    print('macs:', macs / 1000000000)
    print('params:', params / 1000000)

