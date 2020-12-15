# [HS-ResNet](https://arxiv.org/abs/2010.07621)
#### 收录到[PytorchNetHub](https://github.com/bobo0810/PytorchNetHub)

## 说明
- 非官方(官方Paddle版暂未开源)

- 2020.12.15   修复bug，分组数s支持自定义

- 2020.11.16  新增小尺寸输入

- 2020.11.13  参数较ResNet增多，而非减少
  
  >文中式(3)(4)有误,故无法得出HS-ResNet参数及计算量更少的结论
  
  > <img src="https://latex.codecogs.com/gif.latex?param_i&space;=&space;K*K*W*W*(&space;\frac{2^{i-1}-1}{2^{i-1}}&plus;1)_{}^{2}" title="param_i = K*K*W*W*( \frac{2^{i-1}-1}{2^{i-1}}+1)_{}^{2}" />

- 2020.11.12  基于Pytorch官方ResNet，更省内存，支持更多网络

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
# w h>=224
net=hs_resnet50()
# w h<=112
net=hs_resnet50(small_input=True)
pred=net(imgs)
```

## 算法框架
![](https://github.com/bobo0810/HS-ResNet/blob/main/imgs/hs_block.png)