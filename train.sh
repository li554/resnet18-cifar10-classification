#!/usr/bin/bash
# 内置网络
python train.py --lr 0.01 --epoch 100
# 自定义网络
python train.py --lr 0.01 --my_resnet --epoch 100
# 自定义改进网络
python train.py --lr 0.01 --my_improved --epoch 100
# 调整学习率
python train.py --lr 0.1 --my_improved --epoch 100
python train.py --lr 0.001 --my_improved --epoch 100
# 学习率衰减
python train.py --lr 0.01 --my_improved --lr_step 30 --epoch 100
python train.py --lr 0.1 --my_improved --lr_step 30 --epoch 100
python train.py --lr 0.001 --my_improved --lr_step 60 --epoch 100
# 迁移训练
python train.py --lr 0.01 --pretrained --epoch 100
