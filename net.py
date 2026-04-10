from torch import nn
import torch
import torch.nn.functional as F
import math
from torch.nn import Parameter
from attention import TAA

class ARFB(nn.Module):
    def __init__(self, in_ch=64, mid_ch=128, out_ch=128):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, mid_ch, kernel_size=1),nn.BatchNorm2d(mid_ch),nn.PReLU(mid_ch))  # 先不要stride

        self.branch_3x3 = nn.Sequential(nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1,groups=mid_ch))

        self.branch_asym1 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=(1, 3), stride=1, padding=(0, 1), groups=mid_ch),
            )

        self.branch_asym3 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=(3,1), stride=1, padding=(1,0),groups=mid_ch))

        self.bnrule  =nn.Sequential(nn.BatchNorm2d(mid_ch*2),nn.PReLU(mid_ch*2))

        self.conv_out = nn.Sequential(nn.Conv2d(mid_ch*2, out_ch, kernel_size=1),
                                      nn.BatchNorm2d(out_ch))

    def forward(self, x):
        x0 = self.conv1(x)  # 降维
        x2 = self.branch_3x3(x0)
        # 两个非对称卷积分支
        x1 = self.branch_asym1(x0)
        x3 = self.branch_asym3(x0)
        # 标准 3×3 分支
        x_asy1 = x2+x1
        x_asy3 = x3+x2
        # 通道拼接
        x_cat = torch.cat([x_asy1, x_asy3], dim=1)
        x_cat = self.bnrule(x_cat)
        # 最终融合
        return x + self.conv_out(x_cat)

class Bottleneck1(nn.Module):
    def __init__(self, inp=64, oup=128, stride=1, expansion=2):
        super(Bottleneck1, self).__init__()
        self.connect = stride == 1 and inp == oup
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),

            # dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),

            # pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        return self.conv(x)

class bottles0(nn.Module):
    def __init__(self):
        super(bottles0, self).__init__()
        self.inplanes = 64

        self.layer1 = nn.Sequential(
            # AsymBottleneck0(64,128,128),
            Bottleneck1(64, 128, 2, 2),
            ARFB(128,256,128),
        )
        self.layer2 = nn.Sequential(
            Bottleneck1(128, 128, 2, 4),
            ARFB(128,512,128,),
        )
        self.layer3 = nn.Sequential(
            Bottleneck1(128, 128, 2, 4),
            ARFB(128,512,128),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x



class ConvBlock(nn.Module):   #  两种卷积模式，dw决定是否不跨通道卷积和跨通道卷积
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)  #groups不等于1，涉及没有跨通道的权重共享
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)

Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 128, 2, 2],
    [4, 128, 2, 2],
    [4, 128, 2, 2],
]

class MBAF(nn.Module):
    def __init__(self, in_ch=4, mid_ch=32,finanl=64, out_ch=64):
        super().__init__()
        # Step1: 降维
        self.conv1 = nn.Conv2d(4, 32, kernel_size=1)  # 先不要stride

        self.branch_3x3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        # Step2: 非对称卷积分支1
        self.conv2 = nn.Conv2d(4, 64, kernel_size=1)
        self.branch_asym1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1,3), stride=(1,2), padding=(0,1),groups=64),
            nn.Conv2d(64, 64, kernel_size=(3,1), stride=(2,1), padding=(1,0),groups=64))

        self.conv_out = nn.Conv2d(128, 64, kernel_size=1)

    def forward(self, x):
        x0 = self.conv1(x)  # 降维
        x3 = self.branch_3x3(x0)
        # 两个非对称卷积分支
        x1 = self.conv2(x)
        x1 = self.branch_asym1(x1)
        x_cat = torch.cat([x1, x3], dim=1)
        # 最终融合
        return self.conv_out(x_cat)

class AsymBottleneck2(nn.Module):
    def __init__(self, in_ch=64, mid_ch=128, out_ch=128):
        super().__init__()
        # Step1: 降维
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, mid_ch, kernel_size=1),nn.BatchNorm2d(mid_ch),nn.PReLU(mid_ch))  # 先不要stride

        self.branch_3x3 = nn.Sequential(nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1,groups=mid_ch))

        self.branch_asym1 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=(1, 3), stride=1, padding=(0, 1), groups=mid_ch),
            )

        self.branch_asym3 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=(3,1), stride=1, padding=(1,0),groups=mid_ch))

        self.bnrule  =nn.Sequential(nn.BatchNorm2d(mid_ch*2),nn.PReLU(mid_ch*2))

        self.conv_out = nn.Sequential(nn.Conv2d(mid_ch*2, out_ch, kernel_size=3,stride=1,padding=1),
                                      nn.BatchNorm2d(out_ch))

    def forward(self, x):
        x0 = self.conv1(x)
        x2 = self.branch_3x3(x0)
        x1 = self.branch_asym1(x0)
        x3 = self.branch_asym3(x0)
        x_asy1 = x2+x1
        x_asy3 = x3+x2
        x_cat = torch.cat([x_asy1, x_asy3], dim=1)
        x_cat = self.bnrule(x_cat)
        return self.conv_out(x_cat)


class MobileFaceNet(nn.Module):    # 论文中的分类器
    def __init__(self,
                 num_class,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(MobileFaceNet, self).__init__()
        self.conv1 = MBAF()
        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)  #
        self.blocks = bottles0()
        self.conv2 = ConvBlock(bottleneck_setting[-1][1], 512, 1, 1, 0)  # 128 512 2 2
        # 20(10), 4(2), 8(4)
        self.linear7 = ConvBlock(512, 512, (8, 20), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)

        self.fc_out = nn.Linear(128, num_class)  # 输出层了
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x, label):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)  # (2,128,1,1)
        feature = x.view(x.size(0), -1)   # （batch，128）
        out = self.fc_out(feature)  # （batch，10）

        return out, feature


def normalize(x):
    mean = x.mean(dim=-1, keepdim=True)  # 按时间帧平均
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + 1e-8)


class GETE(nn.Module):
    def __init__(self, conv_channels=16, kernel_size=5):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(conv_channels)
        self.relu = nn.ReLU()
        self.expand = nn.Conv1d(conv_channels, 128, kernel_size=1)  # 输出128通道匹配原mel维度
        self.expand_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),  # 局部建模
            # nn.LayerNorm(313),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),  # 投影到128
        )

    def forward(self, x):
        # x: (B, C, 128, 313)
        B, C, F, T = x.shape

        energy = x.sum(dim=2, keepdim=True)

        energy = energy.view(B * C, 1, T)

        trend = self.expand_conv(energy)

        trend = trend.view(B, C, 128, T)
        return trend


class GETE2(nn.Module):
    def __init__(self, out_freq_bins=128, time_frames=313):
        super().__init__()
        self.out_freq_bins = out_freq_bins
        self.time_frames = time_frames

        # 将 [B, 1, 313] → [B, 128, 313]
        self.expand_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),  # 局部建模
            # nn.LayerNorm(313),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, out_freq_bins, kernel_size=1),  # 投影到128
        )

    def forward(self, x):
        x = self.expand_conv(x)          # [B, 128, 313]
        x = x.unsqueeze(1)               # [B, 1, 128, 313]
        return x

class ResidualDilatedBlock3(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(313),  # 归一化时间维度
            nn.PReLU(),
            nn.Conv1d(channels, channels, 3,
                      stride=1,
                      padding=dilation,
                      dilation=dilation,
                      bias=False),
            nn.PReLU(),
            nn.Conv1d(channels, channels, 1, bias=False)  # 融合
        )

    def forward(self, x):
        return self.block(x)

class MTCMM(nn.Module):
    def __init__(self, mel_bins=128, win_lens=[256, 512, 1024, 2048], hop_len=512, num_layer=3):
        super().__init__()
        self.multi_conv = nn.ModuleList([
            nn.Conv1d(1, mel_bins // len(win_lens),
                      kernel_size=w,
                      stride=hop_len,
                      padding=w // 2,
                      bias=False)
            for w in win_lens
        ])
        dilations = [1, 2, 4] if num_layer >= 3 else [1]*num_layer
        self.res_blocks = nn.Sequential(
            *[ResidualDilatedBlock3(mel_bins, d) for d in dilations[:num_layer]]
        )
    def forward(self, x):  # x: (B, 1, 160000)

        feats = [conv(x) for conv in self.multi_conv]  # [(B, mel_bins/4, T), ...]
        out = torch.cat(feats, dim=1)  # (B, mel_bins, 313)

        out = self.res_blocks(out)     # (B, 128, 313)
        return out




class STgramMFN(nn.Module):  #
    def __init__(self, num_classes,
                 c_dim=128,
                 win_len=1024,
                 hop_len=512,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting,
                 use_arcface=False, m=0.5, s=30, sub=1):
        super(STgramMFN, self).__init__()
        self.arcface = ArcMarginProduct(in_features=128, out_features=num_classes,
                                        m=m, s=s, sub=sub) if use_arcface else use_arcface
        self.tgramnet = MTCMM()
        self.cent = GETE2()
        self.energy = GETE()
        self.aten = TAA()
        self.mobilefacenet = MobileFaceNet(num_class=num_classes,
                                           bottleneck_setting=bottleneck_setting)  # 分类器


    def get_tgram(self, x_wav):
        return self.tgramnet(x_wav)

    def forward(self, x_wavs, x_mels, centroids, energys,label=None): #
        x_wavs, x_mels, centroids, energys = x_wavs.unsqueeze(1), x_mels,centroids.unsqueeze(1), energys.unsqueeze(1)#

        x_t = self.tgramnet(x_wavs).unsqueeze(1)    #
        x_conteri = self.cent(centroids)  #
        x_nergy = self.energy(energys)

        x_mel2 = self.aten(x_mels).unsqueeze(1)
        x_mel2 = normalize(x_mel2)



        x = torch.cat((x_conteri,x_nergy,x_mel2,x_t), dim=1)   #
        out, feature = self.mobilefacenet(x, label)  #
        if self.arcface:
            out = self.arcface(feature, label)

        return out, feature


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=10, s=32.0, m=0.50, sub=1, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.sub = sub
        self.weight = Parameter(torch.Tensor(out_features * sub, in_features)) # ）
        nn.init.xavier_uniform_(self.weight)  # 用
        # init.kaiming_uniform_()
        # self.weight.data.normal_(std=0.001)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))  #
        if self.sub > 1:
            cosine = cosine.view(-1, self.out_features, self.sub)
            cosine, _ = torch.max(cosine, dim=2)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=x.device)
        # print(x.device, label.device, one_hot.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return output



if __name__ == '__main__':

    net = STgramMFN(
        num_classes=10,
        use_arcface=True,
        win_len=1024,
        hop_len=512
    )
