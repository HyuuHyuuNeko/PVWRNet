import torch
import torch.nn as nn
import torch.nn.functional as F
from torchstat import stat
from module.attention import ChannelAttention, SpatialAttention, CoordAtt


# ----------------------------------------------------------------------------------------------------
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size,
                     padding=(kernel_size // 2), bias=bias, stride=stride)


# ----------------------------------------------------------------------------------------------------
# Residual Channel Attention Block
class ResCAB(nn.Module):

    def __init__(self, n_feat, kernel_size, reduction=16, bias=False):
        super(ResCAB, self).__init__()
        self.skip_conv1 = conv(n_feat, n_feat, 1, bias=bias)
        self.skip_conv2 = conv(n_feat, n_feat, 3, bias=bias)
        self.ConvBlock = nn.Sequential(
            conv(n_feat, n_feat, kernel_size, bias=bias),
            nn.Hardswish(),
            conv(n_feat, n_feat, kernel_size, bias=bias),
        )
        self.CA = ChannelAttention(n_feat, reduction)
        self.act = nn.Hardswish()

    def forward(self, x):
        skip_1 = self.skip_conv1(x)
        skip_2 = self.skip_conv2(x)
        res = self.ConvBlock(x)
        res = res + skip_1 + skip_2
        res = self.CA(res) * res
        res += x
        return self.act(res)


# ----------------------------------------------------------------------------------------------------
# Encode-Decode Conv Block
class EDConvBlock(nn.Module):

    def __init__(self, n_feat, kernel_size=3, reduction=16, bias=False):
        super(EDConvBlock, self).__init__()
        self.skip_conv1 = conv(n_feat, n_feat, 1, bias=bias)
        self.body = nn.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias),
                                  nn.PReLU(),
                                  conv(n_feat, n_feat, kernel_size, bias=bias))
        self.CA = ChannelAttention(n_feat, reduction)

    def forward(self, x):
        skip = self.skip_conv1(x)
        res = self.body(x)
        res = self.CA(res) * res
        res += skip
        return res


# ----------------------------------------------------------------------------------------------------
# Double Attention Module
class DAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(DAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.cooa = CoordAtt(n_feat, n_feat, 16)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        x1 = self.cooa(x1)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


# ----------------------------------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, scale_unetfeats):
        super(Encoder, self).__init__()

        self.encoder_1 = nn.Sequential(EDConvBlock(n_feat, kernel_size, reduction, bias),
                                       EDConvBlock(n_feat, kernel_size, reduction, bias))
        self.encoder_2 = nn.Sequential(EDConvBlock(n_feat + scale_unetfeats, kernel_size, reduction, bias),
                                       EDConvBlock(n_feat + scale_unetfeats, kernel_size, reduction, bias))
        self.encoder_3 = nn.Sequential(EDConvBlock(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias),
                                       EDConvBlock(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias))

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, x):
        enc1 = self.encoder_1(x)
        x = self.down12(enc1)
        enc2 = self.encoder_2(x)
        x = self.down23(enc2)
        enc3 = self.encoder_3(x)

        return [enc1, enc2, enc3]


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, scale_unetfeats):
        super(Decoder, self).__init__()

        self.decoder_1 = nn.Sequential(EDConvBlock(n_feat, kernel_size, reduction, bias),
                                       EDConvBlock(n_feat, kernel_size, reduction, bias))
        self.decoder_2 = nn.Sequential(EDConvBlock(n_feat + scale_unetfeats, kernel_size, reduction, bias),
                                       EDConvBlock(n_feat + scale_unetfeats, kernel_size, reduction, bias))
        self.decoder_3 = nn.Sequential(EDConvBlock(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias),
                                       EDConvBlock(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias))

        self.skip_1 = EDConvBlock(n_feat, kernel_size, reduction, bias)
        self.skip_2 = EDConvBlock(n_feat + scale_unetfeats, kernel_size, reduction, bias)

        self.spatt = SpatialAttention()

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, in_feature):
        enc1, enc2, enc3 = in_feature
        enc3 = self.spatt(enc3) * enc3
        dec3 = self.decoder_3(enc3)

        x = self.up32(dec3, self.skip_2(enc2))
        dec2 = self.decoder_2(x)

        x = self.up21(dec2, self.skip_1(enc1))
        dec1 = self.decoder_1(x)

        return [dec1, dec2, dec3]


# ----------------------------------------------------------------------------------------------------
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        diffY = y.size()[2] - x.size()[2]
        diffX = y.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = x + y
        return x


# ----------------------------------------------------------------------------------------------------
class PWRNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=64, scale_unetfeats=32, kernel_size=3,
                 reduction=16, bias=False):
        super(PWRNet, self).__init__()

        act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias),
                                           EDConvBlock(n_feat, kernel_size, reduction, bias))
        self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias),
                                           EDConvBlock(n_feat, kernel_size, reduction, bias))
        self.shallow_feat3 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias),
                                           EDConvBlock(n_feat, kernel_size, reduction, bias))

        self.encoderI = Encoder(n_feat, kernel_size, reduction, bias, scale_unetfeats)
        self.encoderII = Encoder(n_feat, kernel_size, reduction, bias, scale_unetfeats)

        self.decoderI = Decoder(n_feat, kernel_size, reduction, bias, scale_unetfeats)
        self.decoderII = Decoder(n_feat, kernel_size, reduction, bias, scale_unetfeats)

        self.dam1 = DAM(n_feat, kernel_size=1, bias=bias)
        self.dam2 = DAM(n_feat, kernel_size=1, bias=bias)
        self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias)

        self.res_rescab1 = nn.Sequential(*[ResCAB(n_feat, 3) for _ in range(6)])
        self.res_rescab2 = nn.Sequential(*[ResCAB(n_feat, 3) for _ in range(4)])
        self.res_rescab3 = nn.Sequential(*[ResCAB(n_feat, 3) for _ in range(2)])

        self.tail = conv(n_feat, out_c, kernel_size, bias=bias)

    def forward(self, input_img):
        H = input_img.size(2)
        W = input_img.size(3)
        # --------------------------------------------------
        # Quartering
        x2LTop = input_img[:, :, 0:int(H / 2), 0:int(W / 2)]
        x2RTop = input_img[:, :, 0:int(H / 2), int(W / 2):W]
        x2LBot = input_img[:, :, int(H / 2):H, 0:int(W / 2)]
        x2RBot = input_img[:, :, int(H / 2):H, int(W / 2):W]
        # --------------------------------------------------
        # Octave
        x1LTop_x2LTop = x2LTop[:, :, 0:int(H / 4), 0:int(W / 4)]
        x1RTop_x2LTop = x2LTop[:, :, 0:int(H / 4), int(W / 4):W]
        x1LBot_x2LTop = x2LTop[:, :, int(H / 4):H, 0:int(W / 4)]
        x1RBot_x2LTop = x2LTop[:, :, int(H / 4):H, int(W / 4):W]

        x1LTop_x2RTop = x2RTop[:, :, 0:int(H / 4), 0:int(W / 4)]
        x1RTop_x2RTop = x2RTop[:, :, 0:int(H / 4), int(W / 4):W]
        x1LBot_x2RTop = x2RTop[:, :, int(H / 4):H, 0:int(W / 4)]
        x1RBot_x2RTop = x2RTop[:, :, int(H / 4):H, int(W / 4):W]

        x1LTop_x2LBot = x2LBot[:, :, 0:int(H / 4), 0:int(W / 4)]
        x1RTop_x2LBot = x2LBot[:, :, 0:int(H / 4), int(W / 4):W]
        x1LBot_x2LBot = x2LBot[:, :, int(H / 4):H, 0:int(W / 4)]
        x1RBot_x2LBot = x2LBot[:, :, int(H / 4):H, int(W / 4):W]

        x1LTop_x2RBot = x2RBot[:, :, 0:int(H / 4), 0:int(W / 4)]
        x1RTop_x2RBot = x2RBot[:, :, 0:int(H / 4), int(W / 4):W]
        x1LBot_x2RBot = x2RBot[:, :, int(H / 4):H, 0:int(W / 4)]
        x1RBot_x2RBot = x2RBot[:, :, int(H / 4):H, int(W / 4):W]

        # --------------------------------------------------
        # main branch
        x1LTop_x2LTop = self.shallow_feat1(x1LTop_x2LTop)
        x1RTop_x2LTop = self.shallow_feat1(x1RTop_x2LTop)
        x1LBot_x2LTop = self.shallow_feat1(x1LBot_x2LTop)
        x1RBot_x2LTop = self.shallow_feat1(x1RBot_x2LTop)

        x1LTop_x2RTop = self.shallow_feat1(x1LTop_x2RTop)
        x1RTop_x2RTop = self.shallow_feat1(x1RTop_x2RTop)
        x1LBot_x2RTop = self.shallow_feat1(x1LBot_x2RTop)
        x1RBot_x2RTop = self.shallow_feat1(x1RBot_x2RTop)

        x1LTop_x2LBot = self.shallow_feat1(x1LTop_x2LBot)
        x1RTop_x2LBot = self.shallow_feat1(x1RTop_x2LBot)
        x1LBot_x2LBot = self.shallow_feat1(x1LBot_x2LBot)
        x1RBot_x2LBot = self.shallow_feat1(x1RBot_x2LBot)

        x1LTop_x2RBot = self.shallow_feat1(x1LTop_x2RBot)
        x1RTop_x2RBot = self.shallow_feat1(x1RTop_x2RBot)
        x1LBot_x2RBot = self.shallow_feat1(x1LBot_x2RBot)
        x1RBot_x2RBot = self.shallow_feat1(x1RBot_x2RBot)

        # --------------------------------------------------
        # into encoder
        x1LTop_x2LTop = self.encoderI(x1LTop_x2LTop)
        x1RTop_x2LTop = self.encoderI(x1RTop_x2LTop)
        x1LBot_x2LTop = self.encoderI(x1LBot_x2LTop)
        x1RBot_x2LTop = self.encoderI(x1RBot_x2LTop)

        x1LTop_x2RTop = self.encoderI(x1LTop_x2RTop)
        x1RTop_x2RTop = self.encoderI(x1RTop_x2RTop)
        x1LBot_x2RTop = self.encoderI(x1LBot_x2RTop)
        x1RBot_x2RTop = self.encoderI(x1RBot_x2RTop)

        x1LTop_x2LBot = self.encoderI(x1LTop_x2LBot)
        x1RTop_x2LBot = self.encoderI(x1RTop_x2LBot)
        x1LBot_x2LBot = self.encoderI(x1LBot_x2LBot)
        x1RBot_x2LBot = self.encoderI(x1RBot_x2LBot)

        x1LTop_x2RBot = self.encoderI(x1LTop_x2RBot)
        x1RTop_x2RBot = self.encoderI(x1RTop_x2RBot)
        x1LBot_x2RBot = self.encoderI(x1LBot_x2RBot)
        x1RBot_x2RBot = self.encoderI(x1RBot_x2RBot)

        # --------------------------------------------------
        # into decoder
        x1LTop_x2LTop = self.decoderI(x1LTop_x2LTop)
        x1RTop_x2LTop = self.decoderI(x1RTop_x2LTop)
        x1LBot_x2LTop = self.decoderI(x1LBot_x2LTop)
        x1RBot_x2LTop = self.decoderI(x1RBot_x2LTop)

        x1LTop_x2RTop = self.decoderI(x1LTop_x2RTop)
        x1RTop_x2RTop = self.decoderI(x1RTop_x2RTop)
        x1LBot_x2RTop = self.decoderI(x1LBot_x2RTop)
        x1RBot_x2RTop = self.decoderI(x1RBot_x2RTop)

        x1LTop_x2LBot = self.decoderI(x1LTop_x2LBot)
        x1RTop_x2LBot = self.decoderI(x1RTop_x2LBot)
        x1LBot_x2LBot = self.decoderI(x1LBot_x2LBot)
        x1RBot_x2LBot = self.decoderI(x1RBot_x2LBot)

        x1LTop_x2RBot = self.decoderI(x1LTop_x2RBot)
        x1RTop_x2RBot = self.decoderI(x1RTop_x2RBot)
        x1LBot_x2RBot = self.decoderI(x1LBot_x2RBot)
        x1RBot_x2RBot = self.decoderI(x1RBot_x2RBot)

        # --------------------------------------------------
        # Concatenate
        x1Top_x2LTop = torch.cat([x1LTop_x2LTop[0], x1RTop_x2LTop[0]], 3)
        x1Bot_x2LTop = torch.cat([x1LBot_x2LTop[0], x1RBot_x2LTop[0]], 3)
        x1_x2LTop = torch.cat([x1Top_x2LTop, x1Bot_x2LTop], 2)

        x1Top_x2RTop = torch.cat([x1LTop_x2RTop[0], x1RTop_x2RTop[0]], 3)
        x1Bot_x2RTop = torch.cat([x1LBot_x2RTop[0], x1RBot_x2RTop[0]], 3)
        x1_x2RTop = torch.cat([x1Top_x2RTop, x1Bot_x2RTop], 2)

        x1Top_x2LBot = torch.cat([x1LTop_x2LBot[0], x1RTop_x2LBot[0]], 3)
        x1Bot_x2LBot = torch.cat([x1LBot_x2LBot[0], x1RBot_x2LBot[0]], 3)
        x1_x2LBot = torch.cat([x1Top_x2LBot, x1Bot_x2LBot], 2)

        x1Top_x2RBot = torch.cat([x1LTop_x2RBot[0], x1RTop_x2RBot[0]], 3)
        x1Bot_x2RBot = torch.cat([x1LBot_x2RBot[0], x1RBot_x2RBot[0]], 3)
        x1_x2RBot = torch.cat([x1Top_x2RBot, x1Bot_x2RBot], 2)

        # --------------------------------------------------
        # ResCAB
        x1LTop_x2LTop = self.res_rescab1(x1_x2LTop)
        x1RTop_x2LTop = self.res_rescab1(x1_x2RTop)
        x1LBot_x2LTop = self.res_rescab1(x1_x2LBot)
        x1RBot_x2LTop = self.res_rescab1(x1_x2RBot)

        # --------------------------------------------------
        # DAM
        x1LTop_x2LTop_samfeats, stage1_img_x1LTop_x2LTop = self.dam1(x1LTop_x2LTop, x2LTop)
        x1RTop_x2LTop_samfeats, stage1_img_x1RTop_x2LTop = self.dam1(x1RTop_x2LTop, x2RTop)
        x1LBot_x2LTop_samfeats, stage1_img_x1LBot_x2LTop = self.dam1(x1LBot_x2LTop, x2LBot)
        x1RBot_x2LTop_samfeats, stage1_img_x1RBot_x2LTop = self.dam1(x1RBot_x2LTop, x2RBot)

        stage1_img_1 = torch.cat([stage1_img_x1LTop_x2LTop, stage1_img_x1RTop_x2LTop], 3)
        stage1_img_2 = torch.cat([stage1_img_x1LBot_x2LTop, stage1_img_x1RBot_x2LTop], 3)
        img1 = torch.cat([stage1_img_1, stage1_img_2], 2)

        # --------------------------------------------------
        # sub branch UP
        x2LTop = self.shallow_feat2(x2LTop)
        x2RTop = self.shallow_feat2(x2RTop)
        x2LBot = self.shallow_feat2(x2LBot)
        x2RBot = self.shallow_feat2(x2RBot)

        x2LTop = x2LTop + x1LTop_x2LTop_samfeats
        x2RTop = x2RTop + x1RTop_x2LTop_samfeats
        x2LBot = x2LBot + x1LBot_x2LTop_samfeats
        x2RBot = x2RBot + x1RBot_x2LTop_samfeats

        # --------------------------------------------------
        # into encoder
        x2LTop = self.encoderII(x2LTop)
        x2RTop = self.encoderII(x2RTop)
        x2LBot = self.encoderII(x2LBot)
        x2RBot = self.encoderII(x2RBot)

        # --------------------------------------------------
        # into decoder
        x2LTop = self.decoderII(x2LTop)
        x2RTop = self.decoderII(x2RTop)
        x2LBot = self.decoderII(x2LBot)
        x2RBot = self.decoderII(x2RBot)

        # --------------------------------------------------
        # Concatenate
        x2Top = torch.cat([x2LTop[0], x2RTop[0]], 3)
        x2Bot = torch.cat([x2LBot[0], x2RBot[0]], 3)
        x2_x1 = torch.cat([x2Top, x2Bot], 2)
        # --------------------------------------------------
        # ResCAB
        x2_x1 = self.res_rescab2(x2_x1)
        # --------------------------------------------------
        # DAM
        x3_samfeats, img2 = self.dam2(x2_x1, input_img)
        # --------------------------------------------------
        # sub branch DOWN
        x3 = self.shallow_feat3(input_img)
        x3_samfeats = x3_samfeats + x3
        # --------------------------------------------------
        # ResCAB
        x3_samfeats = self.res_rescab3(x3_samfeats)
        img3 = self.tail(x3_samfeats)

        return [img3 + input_img, img2, img1]


if __name__ == '__main__':
    # def get_parameter_number(net):
    #     total_num = sum(p.numel() for p in net.parameters())
    #     trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    #     return {'Total': total_num, 'Trainable': trainable_num}

    net = PWRNet()

    """ 6,877,715 """
    # print(get_parameter_number(net))


    """ 
        Total params: 6,803,987
        Total memory: 4142.13MB
        Total MAdd: 623.05GMAdd
        Total Flops: 311.97GFlops
        Total MemR+W: 6.14GB
    """
    # stat(net, (3, 256, 256))

    # from torchkeras import summary
    # summary(net, input_shape=(3, 256, 256), batch_size=4)


