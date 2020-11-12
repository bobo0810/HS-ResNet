# [HS-ResNet](https://arxiv.org/abs/2010.07621)
#### 收录到[PytorchNetHub](https://github.com/bobo0810/PytorchNetHub)

## 说明
- 非官方(官方Paddle版暂未开源)
- 2020.11.12  基于Pytorch官方ResNet，更省内存，且支持更多网络
- 2020.11.11  [简单版本](https://github.com/bobo0810/HS-ResNet/blob/231da98b98e0568af8a42bd4d36507bec97d4c30/hs_resnet.py)，内存占用稍高

## 环境

| python版本 | pytorch版本 | 系统   |
|------------|-------------|--------|
| 3.6        | 1.6.0       | Ubuntu |

## 支持网络
- hs_resnet50
- hs_resnet101
- hs_resnet152
- hs_resnext50_32x4d
- hs_resnext101_32x8d
- hs_wide_resnet50_2
- hs_wide_resnet101_2

## 使用
```python
import hs_resnet50
pred=hs_resnet50(imgs)
```

## 算法框架
![](https://github.com/bobo0810/HS-ResNet/blob/main/imgs/hs_block.png)