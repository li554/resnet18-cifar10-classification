import json
import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm

from dataset import CifarDataset
from metrics import Metric
from models.resnet import Resnet18


def train(net, n_epoch):
    train_iter = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers)
    val_iter = Data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    net.to(device)
    if os.path.exists('runs/exp'):
        exp_dir = 'runs/exp' + str(len(os.listdir('runs')) + 1)
    else:
        exp_dir = 'runs/exp'
    os.makedirs(exp_dir + "/weights/")
    writer = SummaryWriter(exp_dir)
    # 保存参数配置
    with open(exp_dir + "/hyp.json", mode='w') as f:
        hyp = {
            "lr": args.lr,
            "momentum": args.momentum,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "random_seed": args.random_seed,
            'lr_step': args.lr_step,
            'gamma': args.gamma,
            'img_size': args.img_size
        }
        f.write(json.dumps(hyp))
        f.close()

    # 定义损失函数
    loss = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # 学习率衰减
    if args.lr_step != 0:
        milestones = range(args.lr_step, n_epoch, args.lr_step)
    else:
        milestones = []
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)

    best_acc = 0
    # 开始训练
    for epoch in range(n_epoch):

        net.train()
        for X, y in tqdm(train_iter):
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            error = loss(y_hat, y).sum()
            error.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        writer.add_scalar('train/loss', error.item(), epoch)

        net.eval()
        val_loss, val_acc, val_f1_score, n = .0, .0, .0, 0
        batch_count = 0
        for X, y in val_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            error = loss(y_hat, y).sum()
            val_loss += error.item()
            metric = Metric(y_hat.detach().cpu(), y.detach().cpu())
            val_acc += metric.accuracy()
            val_f1_score += metric.f1_score()
            batch_count += 1
            n += y.shape[0]

        writer.add_scalar('val/loss', val_loss / n, epoch)
        writer.add_scalar('val/acc', val_acc / batch_count, epoch)
        # writer.add_scalar('val/f1-score', val_f1_score / batch_count, epoch)

        print(f"epoch:{epoch} loss:{val_loss / n} acc:{val_acc / batch_count}")
        if (val_acc / batch_count) > max(best_acc, 0.7):
            best_acc = val_acc / batch_count
            with open(exp_dir + "/result.txt", mode='w') as f:
                f.write("best accuracy:" + str(best_acc) + "\n")
                # f.write("f1-score" + str(val_f1_score / batch_count) + "\n")
                f.write("epoch:" + str(epoch))
            torch.save(net.state_dict(), exp_dir + "/weights/best.pth")
    torch.save(net.state_dict(), exp_dir + "/weights/last.pth")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--batch_size', type=int, default=128),
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--lr_step', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--my_resnet', action='store_true')
    parser.add_argument('--my_improved', action='store_true')
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', default=100, type=int)

    args = parser.parse_args()
    if args.device == 'cpu':
        device = torch.device(args.device)
    else:
        device = torch.device('cuda:0')
    # 设置随机数种子
    setup_seed(args.random_seed)

    # 加载数据集
    transform_train = transforms.Compose([
        transforms.Resize(size=args.img_size),
        transforms.RandomCrop(args.img_size, padding=4),
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = CifarDataset(img_dir='dataset/trainImages/', img_label='dataset/trainLabels.csv', train=True,
                                 transform=transform_train)
    val_dataset = CifarDataset(img_dir='dataset/trainImages/', img_label='dataset/trainLabels.csv', train=False,
                               transform=transform_test)
    # 定义模型
    if args.pretrained:
        model = resnet18(pretrained=False, num_classes=10)
        state_dict = torch.load('weights/resnet18-f37072fd.pth')
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        model.load_state_dict(state_dict, strict=False)
    elif args.my_resnet or args.my_improved:
        model = Resnet18(num_classes=10, improved=args.my_improved)
    else:
        model = resnet18(pretrained=False, num_classes=10)
    train(model, n_epoch=args.epochs)
