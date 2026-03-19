---
title: 训练营#S2-1
date: 2026-03-19
category: 训练营
tags: [ML, Gradient Descent, DL]
---

# PyTorch入门范例-鸢尾花分类

## 0.导入依赖


```python
# 文件处理
import os

# 允许从命令行传入参数
import argparse

# 处理运行时环境，查看python版本、获取命令行参数的原始列表等
import sys

# pytorch 核心组件
import torch
import torch.optim as optim
import torch.nn as nn

# 数据处理
from torch.utils.data import DataLoader    
from data_loader import iris_dataload

# 进度条与可视化
from tqdm import tqdm

```

## 1.定义超参数
这是深度学习中定义超参数的标准做法，允许在不修改代码的情况下，通过命令行灵活地调整训练配置。在命令行中更改配置的示例如下:
```bash
# A 默认配置
python test.py

# B 增加批大小，更换学习率
python test.py --batch_size 64 --lr 0.01

# C 更换使用设备
python test.py --device cpu
```


```python
parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=100, help='the number of classes')
parser.add_argument('--epochs', type=int, default=20, help='the number of training epoch')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size for training')
parser.add_argument('--lr', type=float, default=0.005, help='star learning rate')   
parser.add_argument('--data_path', type=str, default="/mnt/d/Codes/GNN/NN/Iris_data.txt") 
parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
opt, unknown = parser.parse_known_args()
```

## 2.初始化神经网络


```python
class NeuralNetwork(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x 
```

## 3.定义训练环境


```python
device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
device
```




    device(type='cuda')



## 4.数据集划分


```python
# 使用定义好的数据加载器加载原数据集数据
custom_dataset = iris_dataload("Iris_data.txt")
# 规定训练集、验证集、测试集的大小并根据大小进行随机划分
train_size = int(len(custom_dataset) * 0.7)
validate_size = int(len(custom_dataset) * 0.2)
test_size = len(custom_dataset) - validate_size - train_size
train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, validate_size, test_size])

# 初始化数据加载器
train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False)
validate_loader = DataLoader(validate_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
print("Training set data size:", len(train_loader)*opt.batch_size, ",Validating set data size:", len(validate_loader), ",Testing set data size:", len(test_loader)) 

```

    150 images were found in the dataset.
    Training set data size: 112 ,Validating set data size: 30 ,Testing set data size: 15


## 5.定义推理过程，返回准确率


```python
def infer(model, dataset, device):
    # 告诉pytorch开始验证步骤，不再更新权重了
    model.eval()
    acc_num = 0.0
    # 切换到“省电模式”，不更新模型，不记录计算过程
    with torch.no_grad():
        for data in dataset:
            datas, labels = data
            # 把数据也发送到gpu
            outputs = model(datas.to(device))
            # [1]表示我们只关心最大概率的下标，dim=1表示横向比较，找同一个样本得分最高的类
            predict_y = torch.max(outputs, dim=1)[1]
            # 统计有多少个True，也就是分类正确的,.item()表示把数值从tensor提取成普通的python数字
            acc_num += torch.eq(predict_y, labels.to(device)).sum().item()
    accuracy = acc_num / len(dataset)
    return accuracy
```

## 6.定义训练、测试、验证过程


```python
def main(args):
    print(args)
    
    model = NeuralNetwork(4, 12, 6, 3).to(device) #模型实例化
    loss_function = nn.CrossEntropyLoss() #定义损失函数
    # 把需要训练的参数拿出来
    pg = [p for p in model.parameters() if p.requires_grad] #定义模型参数
    optimizer = optim.Adam(pg, lr = args.lr) # 定义优化器
    
    # 定义模型权重存放地址
    # getcwd: get current working directory
    # join: 智能拼接当前工作地址和想要创建的文件夹地址
    save_path = os.path.join(os.getcwd(), 'resuls/weights')
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
        
    # 开始训练过程
    for epoch in range(opt.epochs):
        #### TRAIN ####
        # 切换到训练模式
        model.train()
        acc_num = torch.zeros(1).to(device) #用于记录当前正确预测的数量，初始化到gpu上避免在多次计算的过程中来回搬运数据
        sample_num = 0 #用于记录当前迭代中已经计算的样本数
        # sys.stdout进度条输出到控制台, ncols表示进度条占据的宽度
        train_bar = tqdm(train_loader, file=sys.stdout, ncols=100)
        
        for datas in train_bar:
            data, label = datas
            # 最后一位的维度如果是1，就把张量压缩
            label = label.squeeze(-1)
            sample_num += data.shape[0]
            # 把旧的梯度删除，否则会累加
            optimizer.zero_grad()
            outputs = model(data.to(device)) #output_shape: [batch_size, num_classes]
            pred_class = torch.max(outputs, dim=1)[1]
            acc_num += torch.eq(pred_class, label.to(device)).sum()
            
            loss = loss_function(outputs, label.to(device))
            loss.backward()
            optimizer.step()
            
            train_acc = acc_num.item() / sample_num
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, opt.epochs, loss)
        
        #### VALIDATE ####
        val_accurate = infer(model=model, dataset = validate_loader, device=device)
        print('[epoch %d] train_loss: %.3f  train_acc: %.3f  val_accuracy: %.3f' %  (epoch + 1, loss, train_acc, val_accurate))
        torch.save(model.state_dict(), os.path.join(save_path, "IrisNet.pth") )
        
        # 迭代后清空指标
        train_acc = 0.0 
        val_accurate = 0.0
    print("Trainning Process Finished Successfully")
    
    test_accurate = infer(model=model, dataset = test_loader, device=device)
    print(' test_accuracy: %.3f' %  ( test_accurate)) 
```


```python
main(opt)
```

    Namespace(num_classes=100, epochs=20, batch_size=16, lr=0.005, data_path='/mnt/d/Codes/GNN/NN/Iris_data.txt', device='cuda')
    train epoch[1/20] loss:1.029: 100%|██████████████████████████████████| 7/7 [00:00<00:00, 304.34it/s]
    [epoch 1] train_loss: 1.029  train_acc: 0.467  val_accuracy: 0.633
    train epoch[2/20] loss:0.872: 100%|██████████████████████████████████| 7/7 [00:00<00:00, 538.49it/s]
    [epoch 2] train_loss: 0.872  train_acc: 0.705  val_accuracy: 0.633
    train epoch[3/20] loss:0.737: 100%|██████████████████████████████████| 7/7 [00:00<00:00, 389.18it/s]
    [epoch 3] train_loss: 0.737  train_acc: 0.705  val_accuracy: 0.633
    train epoch[4/20] loss:0.624: 100%|██████████████████████████████████| 7/7 [00:00<00:00, 500.11it/s]
    [epoch 4] train_loss: 0.624  train_acc: 0.705  val_accuracy: 0.633
    train epoch[5/20] loss:0.533: 100%|██████████████████████████████████| 7/7 [00:00<00:00, 582.28it/s]
    [epoch 5] train_loss: 0.533  train_acc: 0.724  val_accuracy: 0.633
    train epoch[6/20] loss:0.458: 100%|██████████████████████████████████| 7/7 [00:00<00:00, 559.41it/s]
    [epoch 6] train_loss: 0.458  train_acc: 0.771  val_accuracy: 0.633
    train epoch[7/20] loss:0.383: 100%|██████████████████████████████████| 7/7 [00:00<00:00, 583.46it/s]
    [epoch 7] train_loss: 0.383  train_acc: 0.848  val_accuracy: 0.667
    train epoch[8/20] loss:0.305: 100%|██████████████████████████████████| 7/7 [00:00<00:00, 636.34it/s]
    [epoch 8] train_loss: 0.305  train_acc: 0.886  val_accuracy: 0.800
    train epoch[9/20] loss:0.232: 100%|██████████████████████████████████| 7/7 [00:00<00:00, 538.38it/s]
    [epoch 9] train_loss: 0.232  train_acc: 0.933  val_accuracy: 0.867
    train epoch[10/20] loss:0.173: 100%|█████████████████████████████████| 7/7 [00:00<00:00, 538.35it/s]
    [epoch 10] train_loss: 0.173  train_acc: 0.962  val_accuracy: 0.867
    train epoch[11/20] loss:0.128: 100%|█████████████████████████████████| 7/7 [00:00<00:00, 636.35it/s]
    [epoch 11] train_loss: 0.128  train_acc: 0.971  val_accuracy: 0.900
    train epoch[12/20] loss:0.096: 100%|█████████████████████████████████| 7/7 [00:00<00:00, 583.35it/s]
    [epoch 12] train_loss: 0.096  train_acc: 0.971  val_accuracy: 0.933
    train epoch[13/20] loss:0.073: 100%|█████████████████████████████████| 7/7 [00:00<00:00, 500.01it/s]
    [epoch 13] train_loss: 0.073  train_acc: 0.981  val_accuracy: 0.933
    train epoch[14/20] loss:0.057: 100%|█████████████████████████████████| 7/7 [00:00<00:00, 538.49it/s]
    [epoch 14] train_loss: 0.057  train_acc: 0.981  val_accuracy: 0.933
    train epoch[15/20] loss:0.044: 100%|█████████████████████████████████| 7/7 [00:00<00:00, 583.39it/s]
    [epoch 15] train_loss: 0.044  train_acc: 0.981  val_accuracy: 0.933
    train epoch[16/20] loss:0.035: 100%|█████████████████████████████████| 7/7 [00:00<00:00, 583.38it/s]
    [epoch 16] train_loss: 0.035  train_acc: 0.981  val_accuracy: 0.933
    train epoch[17/20] loss:0.028: 100%|█████████████████████████████████| 7/7 [00:00<00:00, 538.67it/s]
    [epoch 17] train_loss: 0.028  train_acc: 0.981  val_accuracy: 0.933
    train epoch[18/20] loss:0.023: 100%|█████████████████████████████████| 7/7 [00:00<00:00, 583.35it/s]
    [epoch 18] train_loss: 0.023  train_acc: 0.981  val_accuracy: 0.933
    train epoch[19/20] loss:0.019: 100%|█████████████████████████████████| 7/7 [00:00<00:00, 466.57it/s]
    [epoch 19] train_loss: 0.019  train_acc: 0.981  val_accuracy: 0.933
    train epoch[20/20] loss:0.016: 100%|█████████████████████████████████| 7/7 [00:00<00:00, 538.39it/s]
    [epoch 20] train_loss: 0.016  train_acc: 0.990  val_accuracy: 0.933
    Trainning Process Finished Successfully
     test_accuracy: 0.867



```python

```
