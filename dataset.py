from pathlib import Path
import pandas as pd
import torch.utils.data as Data
from PIL import Image
import numpy as np
import torch
import random
from torchvision import transforms

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class CifarDataset(Data.Dataset):
    """初始化数据集"""

    @staticmethod
    def get_label(label_path):
        if label_path is not None:
            df = pd.read_csv(label_path)
            class_dict = {label: i for i, label in enumerate(classes)}
            df['label'] = df['label'].apply(lambda x: class_dict[x])
            return list(df['label'].values)
        else:
            return None

    def __init__(self, img_dir='dataset/trainImages/', train=True, img_label=None, transform=None):
        self.img_path = list(Path(img_dir).glob('*.png'))
        self.img_path.sort(key=lambda x: int(x.name.split('.')[0]))
        self.img_label = self.get_label(img_label)
        if img_label is not None:
            num_train = int(0.8 * len(self.img_path))
            index_list = list(range(len(self.img_path)))
            random.seed(42)
            indexes = random.sample(index_list, num_train)
            if not train:
                indexes = list(set(index_list) - set(indexes))
                # [index for index in index_list if index not in indexes]
            self.img_path = [self.img_path[index] for index in indexes]
            self.img_label = [self.img_label[index] for index in indexes]
        self.transform = transform

        '''根据下标返回数据(img和label)'''

    def __getitem__(self, index):
        if self.img_label is not None:
            img = Image.open(self.img_path[index]).convert('RGB')
            label = np.array(self.img_label[index], dtype=int)
            if self.transform is not None:
                img = self.transform(img)
            return img, torch.from_numpy(label)
        else:
            img = Image.open(self.img_path[index]).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img, torch.from_numpy(np.array([]))

    '''返回数据集长度'''

    def __len__(self):
        return len(self.img_path)

# if __name__ == '__main__':
#     # 加载数据集
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
#     ])
#
#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
#
#     train_dataset = CifarDataset(img_dir='dataset/trainImages/', img_label='dataset/trainLabels.csv', train=True,
#                                  transform=transform_train)
#     val_dataset = CifarDataset(img_dir='dataset/trainImages/', img_label='dataset/trainLabels.csv', train=False,
#                                transform=transform_test)
