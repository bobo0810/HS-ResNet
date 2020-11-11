import torch
import torch.nn as nn
from net.hs_resnet.hs_block import HSBlock

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class HSBottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, s=8):
        super().__init__()
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        #################################################################
        # 特征图尺度改变，用原生ResBlock
        if stride != 1:
            self.conv_3x3 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            out_ch = out_channels
        # 特征图尺度不变，用HSBlock
        else:
            self.conv_3x3 = HSBlock(in_ch=out_channels, s=s)
        #################################################################
        self.conv_1x1_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * HSBottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * HSBottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * HSBottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * HSBottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * HSBottleNeck.expansion)
            )

    def forward(self, x):
        x_hs = self.conv_1x1(x)
        x_hs = self.conv_3x3(x_hs)
        x_hs = self.conv_1x1_2(x_hs)
        return nn.ReLU(inplace=True)(x_hs + self.shortcut(x))


class HS_ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=512):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


def hs_resnet50():
    """ return a ResNet 50 object
    """
    # return HS_ResNet(BottleNeck, [3, 4, 6, 3])
    return HS_ResNet(HSBottleNeck, [3, 4, 6, 3])


def hs_resnet101():
    """ return a ResNet 101 object
    """
    # return HS_ResNet(BottleNeck, [3, 4, 23, 3])
    return HS_ResNet(HSBottleNeck, [3, 4, 23, 3])


def hs_resnet152():
    """ return a ResNet 152 object
    """
    # return HS_ResNet(BottleNeck, [3, 8, 36, 3])
    return HS_ResNet(HSBottleNeck, [3, 8, 36, 3])


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    device = torch.device("cuda:0")
    # [batch,channel,H,W]
    feature = torch.rand(4, 3, 112, 96).to(device)
    model = hs_resnet50().to(device).train()
    result = model(feature)
    print(result.size())
