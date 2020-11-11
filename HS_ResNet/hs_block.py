import os
import torch
import torch.nn as nn


class HSBlock(nn.Module):
    def __init__(self, in_ch, s=8):
        '''
        特征大小不改变
        :param in_ch: 输入通道
        :param s: 分组数
        :param w: 滤波器宽度（卷积核的输出通道）
        '''
        super(HSBlock, self).__init__()
        self.s = s
        self.module_list = nn.ModuleList()
        # 避免无法整除通道数
        in_ch, in_ch_last = (in_ch // s, in_ch // s) if in_ch % s == 0 else (in_ch // s + 1, in_ch % s)
        self.module_list.append(nn.Sequential())
        acc_channels = 0
        for i in range(1,self.s):
            if i == 1:
                channels=in_ch
                acc_channels=channels//2
            elif i == s - 1:
                channels = in_ch_last + acc_channels
            else:
                channels=in_ch+acc_channels
                acc_channels=channels//2
            self.module_list.append(self.conv_bn_relu(in_ch=channels, out_ch=channels))
        self.initialize_weights()

    def conv_bn_relu(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        return conv_bn_relu

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = list(x.chunk(chunks=self.s, dim=1))
        for i in range(1, len(self.module_list)):
            y = self.module_list[i](x[i])
            if i == len(self.module_list) - 1:
                x[0] = torch.cat((x[0], y), 1)
            else:
                y1, y2 = y.chunk(chunks=2, dim=1)
                x[0] = torch.cat((x[0], y1), 1)
                x[i + 1] = torch.cat((x[i + 1], y2), 1)
        return x[0]

# if __name__ == '__main__':
#     os.environ['CUDA_VISIBLE_DEVICES'] = "1"
#     device = torch.device("cpu")
#     # [batch,channel,H,W]
#     feature = torch.rand(6, 40, 24, 24).to(device)
#     in_ch = feature.shape[1]
#
#     # 表2中s=5  论文推荐s=8
#     hs_block = HSBlock(in_ch,s=5).to(device)
#     hs_block = hs_block.train()
#     result = hs_block(feature)
#     print(result.size())
