数据集请前往kaggle官网下载 https://www.kaggle.com/c/cifar-10/data

下载完成后解压放置到dataset文件夹下，目标结构如下

![image-20211230115604811](https://s2.loli.net/2021/12/30/KaNCmsJqVcOAFfE.png)

当然读者亦可使用torchvision内置的cifar10数据集，运行时会先下载cifar10数据集，可能下载比较慢，可以先运行一次，找到链接后自行下载完成后放到dataset文件夹下，然后重新运行

```python
import torch
import torchvision
import torchvision.transforms as transforms

trainset = torchvision.datasets.CIFAR10(root='./dataset/cifar', train=True,
                                        download=True,transform=None)

testset = torchvision.datasets.CIFAR10(root='./dataset/cifar', train=False,
                                       download=True, transform=None)
```

