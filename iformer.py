from timm.models.layers import PatchEmbed, Mlp, DropPath
from timm.models.vision_transformer import Attention, Block

from timm.models.layers import make_divisible

import torch
import torch.nn as nn

config1 = {
    'channels': [48, 96, 192, 320, 384],
    'depths': [3, 3, 9, 3],
    'conv_rate_low': [2/3, 1/2, 1/10, 1/12],
    'conv_rate_high': [2/3, 1/2, 3/10, 1/12],
}
config2 = {
    'channels': [48, 96, 192, 384, 512],
    'depths': [4, 6, 14, 6],
    'conv_rate_low': [2/3, 1/2, 2/12, 1/16],
    'conv_rate_high': [2/3, 1/2, 4/12, 1/16],
}
config3 = {
    'channels': [48, 96, 192, 448, 640],
    'depths': [4, 6, 18, 8],
    'conv_rate_low': [2/3, 1/2, 2/14, 1/20],
    'conv_rate_high': [2/3, 1/2, 4/14, 1/20],
}

class InceptionMixer(nn.Module):
    def __init__(self, input_channels, tran_ratio, pool_stride, img_size):
        super().__init__()
        # have to be even, not odd
        # conv_chan = int(input_channels*conv_ratio/2) * 2
        tran_chans = make_divisible(input_channels * tran_ratio, 32)
        conv_chans = input_channels - tran_chans
        self.high = conv_chans
        self.low  = tran_chans
        self.maxpool_fc = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(self.high // 2, self.high // 2, 1),
            nn.BatchNorm2d(self.high // 2),
            nn.ReLU6(inplace=True),
        )

        self.fc_dw = nn.Sequential(
            nn.Conv2d(self.high // 2, self.high // 2, 1),
            nn.BatchNorm2d(self.high // 2),
            nn.ReLU6(inplace=True),
            nn.Conv2d(self.high // 2, self.high // 2, 3, padding=1, groups=self.high // 2),
            nn.BatchNorm2d(self.high // 2),
            # nn.ReLU6(inplace=True),
        )
        # self.factorized_tran = nn.Conv2d(input_channels, tran_chan, 1)
        # self.factorized_cnn = nn.Conv2d(input_channels, conv_chan, 1)
        
        self.pool_stride = pool_stride
        # self.low should be divided by num_heads
        # print(self.low)
        self.attn = Attention(self.low, num_heads=self.low//32)#8)
        H = W = img_size
        patch_size = int(H // pool_stride * W // pool_stride)
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_size, self.low))

        self.fuse_dw = nn.Conv2d(input_channels, input_channels, 3, padding=1, groups=input_channels)
        self.fuse_linear = nn.Conv2d(input_channels, input_channels, 1)
        # print(conv_chan, input_channels-conv_chan)

    def forward(self, x):
        B, C, H, W = x.shape

        # x = self.split(x)
        # conv = self.factorized_cnn(x)
        # conv1 = conv[:, :self.conv_chan//2, ...]
        # conv2 = conv[:, self.conv_chan//2:self.conv_chan, ...]
        # attn = self.factorized_tran(x)

        X_h1 = x[:, :self.high//2, ...]
        X_h2 = x[:, self.high//2:self.high, ...]
        X_l  = x[:, -self.low:, ...]

        Y_h1 = self.maxpool_fc(X_h1)
        Y_h2 =      self.fc_dw(X_h2)

        Y_l = nn.AdaptiveAvgPool2d((H//self.pool_stride, W//self.pool_stride))(X_l)
        Y_l = Y_l.flatten(2).transpose(1,2)
        Y_l = Y_l + self.pos_embed
        # print(Y_l.shape, self.pos_embed.shape)
        # print(attn.shape)
        Y_l = self.attn(Y_l)
        Y_l = Y_l.reshape(B, -1, H//self.pool_stride, W//self.pool_stride)
        Y_l = nn.UpsamplingBilinear2d((H, W))(Y_l)

        Y_c = torch.cat([Y_l, Y_h1, Y_h2], dim=1)
        out = self.fuse_linear(Y_c + self.fuse_dw(Y_c))
        # print("conv", conv_out1.shape, conv_out2.shape, "attn", attn_out.shape)
        # print(out.shape)
        return out

class iFormerBlock(nn.Module):
    def __init__(
        self, input_channels, tran_ratio, pool_stride, img_size,
        mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
        drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm
        ):
        tran_chan = make_divisible(input_channels * tran_ratio, 8)
        super().__init__()
        dim = input_channels
        self.norm1 = norm_layer(dim)
        self.inceptionmixer = InceptionMixer(input_channels, tran_ratio, pool_stride, img_size)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, C, H, W = x.shape
        # x = x.flatten(2).transpose(1,2)
        # print(x.shape)

        x = x + self.drop_path(
            self.inceptionmixer(
                self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            )
        )
        x = x + self.drop_path(
            self.mlp(
                self.norm2(x.permute(0, 2, 3, 1))
            ).reshape(B, C, H, W)
        )

        # x = x.transpose(1,2).reshape(B, C, H, W)
        return x


class InceptionTransformer(nn.Module):
    def __init__(self, config = config1, img_size=224):
        super().__init__()
        print(config)
        chans = config['channels']
        depth = config['depths']
        conv_ratio_low = config['conv_rate_low']
        conv_ratio_high = config['conv_rate_high']

        self.stage1_patch_embed = nn.Sequential(
            nn.Conv2d(3, chans[0], 3, 2, padding=1),
            nn.Conv2d(chans[0], chans[1], 3, 2, padding=1),
        )
        img_size /= 4
        self.iBlocks1 = nn.Sequential(
            *[iFormerBlock(chans[1], 1-conv_ratio_low[0], 2, img_size) for _ in range(depth[0])]
        )

        self.stage2_patch_embed = nn.Conv2d(chans[1], chans[2], 2, 2)
        img_size /= 2
        self.iBlocks2 = nn.Sequential(
            *[iFormerBlock(chans[2], 1-conv_ratio_low[1], 2, img_size) for _ in range(depth[1])]
        )

        c = [conv_ratio_high[2]-i*(conv_ratio_high[2] - conv_ratio_low[2])/depth[2] for i in range(depth[2])]
        # repeat = make_divisible(depth[2]/3, 1)
        # c = [conv_ratio_high[2]]*repeat + [(conv_ratio_high[2] - conv_ratio_low[2])/2]*(depth[2] - 2*repeat) + [conv_ratio_low[2]]*repeat
        # print(c)
        self.stage3_patch_embed = nn.Conv2d(chans[2], chans[3], 2, 2)
        img_size /= 2
        self.iBlocks3 = nn.Sequential(
            *[iFormerBlock(chans[3], 1-c[i], 1, img_size) for i in range(depth[2])]
        )

        self.stage4_patch_embed = nn.Conv2d(chans[3], chans[4], 2, 2)
        img_size /= 2
        self.iBlocks4 = nn.Sequential(
            *[iFormerBlock(chans[4], 1-conv_ratio_low[3], 1, img_size) for _ in range(depth[3])]
        )
        
    
    def forward(self, x):
        x = self.stage1_patch_embed(x)#.flatten(2).transpose(1,2)
        #print(x.shape)
        x = self.iBlocks1(x)

        x = self.stage2_patch_embed(x)
        x = self.iBlocks2(x)

        x = self.stage3_patch_embed(x)
        x = self.iBlocks3(x)

        x = self.stage4_patch_embed(x)
        x = self.iBlocks4(x)


from thop import profile, clever_format

if __name__ == "__main__":
    # iblk = IBlock(320, 3/10, 7/10, 2, 2)
    # input = torch.rand(1, 320, 64, 64)
    # output = iblk(input)
    
    input = torch.rand(1, 3, 224, 224)
    iFormer = InceptionTransformer(config3)
    # print(sum(p.numel() for p in iFormer.parameters()))
    # output = iFormer(input)

    flops, params = profile(iFormer, inputs=(input, ), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
