#!/usr/bin/env python
# coding: utf-8

import torch
import collections
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import time
from IPython import display
import matplotlib_inline
import torch.nn.functional as F
import os
import torchvision.models as models

class AlexNet(nn.Module):
    """LeNet 模型"""
    def __init__(self, num_outputs,batch_size=256,lr=0.1, device='cuda'):
        super().__init__()
        self.batch_size=batch_size
        self.lr=lr

        self.net = nn.Sequential(
            nn.LazyConv2d(96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.LazyLinear(4096), nn.ReLU(),nn.Dropout(p=0.5),
            nn.LazyLinear(num_outputs))
        # 损失函数
        self.loss_fn = F.cross_entropy
        # 优化器
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        # 移动到设备
        self.device = torch.device(device)
        self.to(self.device)
        self.net.apply(self.init_cnn)
    def forward(self, X):

        return self.net(X)
    
    def apply_init(self, inputs, init=None):
        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)

    def init_cnn(module):  #@save
        """Initialize weights for CNNs."""
        if type(module) == nn.Linear or type(module) == nn.Conv2d:
            nn.init.xavier_uniform_(module.weight)

class VGG(nn.Module):
    """VGG 模型"""
    def __init__(self,arch, num_outputs,batch_size=256,lr=0.1, device='cuda'):
        super().__init__()
        self.batch_size=batch_size
        self.lr=lr

        conv_blks = []
        for (num_convs, out_channels) in arch:
            conv_blks.append(self.vgg_block(num_convs, out_channels))
        self.net = nn.Sequential(
            *conv_blks, nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(num_outputs))
        # 损失函数
        self.loss_fn = F.cross_entropy
        # 优化器
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        # 移动到设备
        self.device = torch.device(device)
        self.to(self.device)
        self.net.apply(self.init_cnn)
    def forward(self, X):

        return self.net(X)

    def vgg_block(self,num_convs, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        return nn.Sequential(*layers)
    
    def apply_init(self, inputs, init=None):
        """延后初始化"""
        if init is not None:
            inputs = [x.to(self.device) for x in inputs]
            # 先做一次前向传播来初始化 Lazy 层
            with torch.no_grad():
                self.forward(*inputs)
            # 然后应用初始化
            self.net.apply(init)

    def init_cnn(self,module):  #@save
        """Initialize weights for CNNs."""
        if type(module) == nn.Linear or type(module) == nn.Conv2d:
            nn.init.xavier_uniform_(module.weight)

class Inception(nn.Module):
    # c1--c4 are the number of output channels for each branch
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)
        # Branch 2
        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)
        # Branch 3
        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
        # Branch 4
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)

    def forward(self, x):
        b1 = F.relu(self.b1_1(x))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        b4 = F.relu(self.b4_2(self.b4_1(x)))
        return torch.cat((b1, b2, b3, b4), dim=1)
    
class GoogleNet(nn.Module):
    """GoogleNet 模型"""
    def __init__(self, num_outputs,batch_size=256,lr=0.1, device='cuda'):
        super(GoogleNet, self).__init__()
        self.batch_size=batch_size
        self.lr=lr
        self.net = nn.Sequential(self.b1(), self.b2(), self.b3(), self.b4(),
                             self.b5(), nn.LazyLinear(num_outputs))
        # 损失函数
        self.loss_fn = F.cross_entropy
        # 优化器
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        # 移动到设备
        self.device = torch.device(device)
        self.to(self.device)
        self.net.apply(self.init_cnn)
    def forward(self, X):

        return self.net(X)

    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    def b2(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=1), nn.ReLU(),
            nn.LazyConv2d(192, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    def b3(self):
        return nn.Sequential(Inception(64, (96, 128), (16, 32), 32),
                            Inception(128, (128, 192), (32, 96), 64),
                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    def b4(self):
        return nn.Sequential(Inception(192, (96, 208), (16, 48), 64),
                            Inception(160, (112, 224), (24, 64), 64),
                            Inception(128, (128, 256), (24, 64), 64),
                            Inception(112, (144, 288), (32, 64), 64),
                            Inception(256, (160, 320), (32, 128), 128),
                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    def b5(self):
        return nn.Sequential(Inception(256, (160, 320), (32, 128), 128),
                            Inception(384, (192, 384), (48, 128), 128),
                            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
    def apply_init(self, inputs, init=None):
        """延后初始化"""
        if init is not None:
            inputs = [x.to(self.device) for x in inputs]
            # 先做一次前向传播来初始化 Lazy 层
            with torch.no_grad():
                self.forward(*inputs)
            # 然后应用初始化
            self.net.apply(init)

    def init_cnn(self,module):  #@save
        """Initialize weights for CNNs."""
        if type(module) == nn.Linear or type(module) == nn.Conv2d:
            nn.init.xavier_uniform_(module.weight)

class Residual(nn.Module): 
    """The Residual block of ResNet models."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1,
                                   stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1,
                                       stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

class ResNet(nn.Module):
    def __init__(self,arch, num_outputs,batch_size=256,lr=0.1, device='cuda'):
            super(ResNet, self).__init__()
            self.batch_size=batch_size
            self.lr=lr
            self.net = nn.Sequential(self.b1())
            for i, b in enumerate(arch):
                self.net.add_module(f'b{i+2}', self.block(*b, first_block=(i==0)))
            self.net.add_module('last', nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                nn.LazyLinear(num_outputs)))
            # 损失函数
            self.loss_fn = F.cross_entropy
            # 优化器
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
            # 移动到设备
            self.device = torch.device(device)
            self.to(self.device)
            self.net.apply(self.init_cnn)

    def forward(self, X):
    
            return self.net(X)
    
    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    def block(self, num_residuals, num_channels, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels))
        return nn.Sequential(*blk)
    def apply_init(self, inputs, init=None):
            """延后初始化"""
            if init is not None:
                inputs = [x.to(self.device) for x in inputs]
                # 先做一次前向传播来初始化 Lazy 层
                with torch.no_grad():
                    self.forward(*inputs)
                # 然后应用初始化
                self.net.apply(init)
    
    def init_cnn(self,module):  #@save
        """Initialize weights for CNNs."""
        if type(module) == nn.Linear or type(module) == nn.Conv2d:
            nn.init.xavier_uniform_(module.weight)

import torchvision.models as models

class PretrainedResNet(nn.Module):
    """使用 torchvision 预训练 ResNet 进行迁移学习"""
    def __init__(self, num_outputs=10, batch_size=128, lr=0.001, 
                  freeze_backbone=True, device='cuda'):
        super(PretrainedResNet, self).__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device(device)
        
        # 加载预训练模型
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 冻结骨干网络（特征提取模式）
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 获取分类头输入维度
        fc_in_features = self.backbone.fc.in_features
        
        # 替换分类头
        self.backbone.fc = nn.Sequential(
            nn.Linear(fc_in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_outputs)
        )
        
        # 损失函数
        self.loss_fn = F.cross_entropy
        
        # 优化器（只优化可训练参数）
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.backbone.parameters()),
            lr=lr
        )
        
        # 移动到设备
        self.to(self.device)
    
    def forward(self, X):
        return self.backbone(X)
    

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class FashionMNISTData:
    """支持不同 batch_size 的数据管理器"""
    def __init__(self, root='./data'):
        self.root = root
        self._load_datasets()
    def _load_datasets(self):
        """加载一次数据集（不创建DataLoader）"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.train_dataset = datasets.FashionMNIST(
            root=self.root, train=True, download=True, transform=transform
        )
        self.test_dataset = datasets.FashionMNIST(
            root=self.root, train=False, download=True, transform=transform
        )
    def dataset_Resize(self,resize,rgb=False):
        if rgb:
            transform = transforms.Compose([
                            transforms.Resize(resize),
                            transforms.Grayscale(num_output_channels=3) ,
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ])
        else:
            transform = transforms.Compose([
                    transforms.Resize(resize),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
        self.train_dataset = datasets.FashionMNIST(
            root=self.root, train=True, download=True, transform=transform
        )
        self.test_dataset = datasets.FashionMNIST(
            root=self.root, train=False, download=True, transform=transform
        )
    def get_train_loader(self, batch_size=256, shuffle=True):
        """创建训练DataLoader（可指定batch_size）"""
        return DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle
        )
    def get_test_loader(self, batch_size=256, shuffle=False):
        """创建测试DataLoader（可指定batch_size）"""
        return DataLoader(
            self.test_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle
        )
    def text_labels(self, indices):
        """Return text labels."""
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [labels[int(i)] for i in indices]
        
    def show_images(self,imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
        """Plot a list of images."""
        figsize = (num_cols * scale, num_rows * scale)
        _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        axes = axes.flatten()
        for i, (ax, img) in enumerate(zip(axes, imgs)):
            try:
                img = img.detach().numpy()
            except:
                pass
            ax.imshow(img)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            if titles:
                ax.set_title(titles[i])
        return axes
        
    def visualize(self, batch, nrows=1, ncols=8, labels=[]):
        X, y = batch
        if not labels:
            labels = self.text_labels(y)
        return self.show_images(X.squeeze(1), nrows, ncols, titles=labels)

class ProgressBoard():  #@save
    """The board that plots data points in animation."""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.xlabel=xlabel
        self.ylabel=ylabel
        self.xlim=xlim
        self.ylim=ylim
        self.xscale=xscale
        self.yscale=yscale
        self.ls=ls
        self.colors=colors
        self.fig=fig
        self.axes=axes
        self.figsize=figsize
        self.display=display

    def draw(self, x, y, label, every_n=1):
        Point = collections.namedtuple('Point', ['x', 'y'])
        if not hasattr(self, 'raw_points'):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        points = self.raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))
        if len(points) != every_n:
            return
        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]),
                            mean([p.y for p in points])))
        points.clear()
        if not self.display:
            return
        matplotlib_inline.backend_inline.set_matplotlib_formats('svg') 
        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize)
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(plt.plot([p.x for p in v], [p.y for p in v],
                                            linestyle=ls, color=color)[0])
            labels.append(k)
        axes = self.axes if self.axes else plt.gca()
        if self.xlim: axes.set_xlim(self.xlim)
        if self.ylim: axes.set_ylim(self.ylim)
        if not self.xlabel: self.xlabel = self.x
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)
        display.display(self.fig)
        display.clear_output(wait=True)

class Trainer:
    """通用训练器类，封装训练、评估等功能"""

    def __init__(
        self,
        model: nn.modules,
        data: FashionMNISTData,
        max_epoch:int,
        verbose: bool = True,
        save_best: bool = True,

    ):
        """
        初始化训练器

        参数:
            model: 模型
            date: 数据
            verbose: 是否打印详细信息
            save_best: 是否保存最佳模型
        """
        self.model = model
        self.date = data
        self.max_epoch=max_epoch
        self.verbose = verbose
        self.save_best = save_best
        self.board=ProgressBoard()

        # 移动到设备
        self.model.to(self.model.device)

        self.train_loader=self.date.get_train_loader(batch_size=self.model.batch_size)
        self.test_loader=self.date.get_test_loader(batch_size=self.model.batch_size)
        # 损失函数
        self.loss_fn = self.model.loss_fn

        # 优化器
        self.optimizer = self.model.optimizer

        # 历史记录
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'lr': [],
            'epoch_time': []
        }

        # 最佳模型追踪
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = -1

        # 训练状态
        self.current_epoch = 0
        self.is_trained = False

    def plot(self, key, value, train):
        """Plot a point in animation."""
        self.board.xlabel = 'epoch'
        plot_train_per_epoch=2
        plot_valid_per_epoch=1
        num_train_batches=len(self.train_loader)
        num_test_batches=len(self.test_loader)
        if train:
            x = self.train_batch_idx / \
                num_train_batches + self.current_epoch - 1
            n = num_train_batches / \
                plot_train_per_epoch
        else:
            x = self.current_epoch
            n = num_test_batches / \
                plot_valid_per_epoch
        self.board.draw(x, value.cpu().detach().numpy(),
                        ('train_' if train else 'test_') + key,
                        every_n=int(n))

    def accuracy(self, Y_hat, Y, averaged=True):
        """Compute the number of correct predictions."""
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        preds = Y_hat.argmax(axis=1).type(Y.dtype)
        compare = (preds == Y.reshape(-1)).type(torch.float32)
        return compare.mean() if averaged else compare

    def train_epoch(self) :
        """
        训练一个epoch

        返回:
            包含平均损失和准确率的字典
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        self.train_batch_idx=0

        for X, y in self.train_loader:
            # 移动到设备
            X, y = X.to(self.model.device), y.to(self.model.device)

            # 前向传播
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)

            self.plot('loss',loss,train=True)
            # self.plot('acc',self.accuracy(y_pred, y),train=True)
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.train_batch_idx+=1

            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(y_pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        # 计算平均指标
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total

        return avg_loss,accuracy

    def evaluate(self) :
        """
        评估模型

        返回:
            包含平均损失和准确率的字典
        """

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.model.device), y.to(self.model.device)
                y_pred = self.model(X)
                loss = self.loss_fn(y_pred, y)

                self.plot('loss',loss,train=False)
                self.plot('acc',self.accuracy(y_pred, y),train=False)

                total_loss += loss.item()
                _, predicted = torch.max(y_pred, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        avg_loss = total_loss / len(self.test_loader)
        accuracy = correct / total


        return avg_loss,accuracy

    def print_last_epoch(self,name):
        """打印最后一个epoch的训练结果"""
        history=self.history
        if not history['train_loss']:
            print("暂无训练数据！")
            return
        
        last_idx = -1  # Python中-1表示最后一个元素
        
        print("=" * 60)
        print(f"{name:^60}")
        print("=" * 60)
        print(f"{'指标':<20} {'值':>30}")
        print("-" * 60)
        print(f"{'Train Loss':<20} {history['train_loss'][last_idx]:>30.4f}")
        print(f"{'Train Accuracy':<20} {history['train_acc'][last_idx]:>30.4f}")
        print(f"{'Test Loss':<20} {history['test_loss'][last_idx]:>30.4f}")
        print(f"{'Test Accuracy':<20} {history['test_acc'][last_idx]:>30.4f}")
        print("=" * 60)

    def print_all_epochs(self, name):
        """打印所有轮次的训练结果"""
        history = self.history
        if not history['train_loss']:
            print("暂无训练数据！")
            return
        
        num_epochs = len(history['train_loss'])
        
        # 打印标题
        print("=" * 100)
        print(f"{name:^100}")
        print("=" * 100)
        print(f"{'Epoch':<8} {'Train Loss':<15} {'Train Acc':<15} {'Test Loss':<15} {'Test Acc':<15} {'Time(s)':<12}")
        print("-" * 100)
        
        # 打印每个epoch的数据
        for i in range(num_epochs):
            print(f"{i+1:<8} "
                f"{history['train_loss'][i]:<15.4f} "
                f"{history['train_acc'][i]:<15.4f} "
                f"{history['test_loss'][i]:<15.4f} "
                f"{history['test_acc'][i]:<15.4f} "
                f"{history['epoch_time'][i]:<12.2f}")
        
        print("=" * 100)
 
    def train_AlexNet(
        self,
        plot_curves: bool = True
    ):
        """
        完整训练流程

        参数:
            num_epochs: 训练轮数
            plot_curves: 是否绘制训练曲线

        返回:
            训练历史记录
        """
        start_time = time.time()
        num_epochs=self.max_epoch

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # 1. 训练一个epoch
            train_loss, train_accuracy = self.train_epoch()

            # 2. 记录训练指标
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_accuracy)


            test_loss, test_accuracy = self.evaluate()
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_accuracy)

            # 5. 记录耗时
            epoch_time = time.time() - epoch_start_time
            self.history['epoch_time'].append(epoch_time)

        self.print_last_epoch(name='AlexNet')
        fig = self.board.fig  # 获取当前图形
        fig.suptitle(f'batch_sizes={self.model.batch_size}_lr={self.model.lr}')
        display.display(fig)  # 显示图形
        display.clear_output(wait=True)
        return fig

    def train_VGG(
        self,
        plot_curves: bool = True
    ):
        """
        完整训练流程

        参数:
            num_epochs: 训练轮数
            plot_curves: 是否绘制训练曲线

        返回:
            训练历史记录
        """
        start_time = time.time()
        num_epochs=self.max_epoch

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # 1. 训练一个epoch
            train_loss, train_accuracy = self.train_epoch()

            # 2. 记录训练指标
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_accuracy)


            test_loss, test_accuracy = self.evaluate()
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_accuracy)

            # 5. 记录耗时
            epoch_time = time.time() - epoch_start_time
            self.history['epoch_time'].append(epoch_time)

        self.print_last_epoch(name='VGC')
        fig = self.board.fig  # 获取当前图形
        fig.suptitle(f'batch_sizes={self.model.batch_size}_lr={self.model.lr}')
        display.display(fig)  # 显示图形
        display.clear_output(wait=True)
        return fig

    def train_GoogleNet(
        self,
        plot_curves: bool = True
    ):
        """
        完整训练流程

        参数:
            num_epochs: 训练轮数
            plot_curves: 是否绘制训练曲线

        返回:
            训练历史记录
        """
        start_time = time.time()
        num_epochs=self.max_epoch

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # 1. 训练一个epoch
            train_loss, train_accuracy = self.train_epoch()

            # 2. 记录训练指标
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_accuracy)


            test_loss, test_accuracy = self.evaluate()
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_accuracy)

            # 5. 记录耗时
            epoch_time = time.time() - epoch_start_time
            self.history['epoch_time'].append(epoch_time)

        self.print_last_epoch(name='GoogleNet')
        fig = self.board.fig  # 获取当前图形
        fig.suptitle(f'batch_sizes={self.model.batch_size}_lr={self.model.lr}')
        display.display(fig)  # 显示图形
        display.clear_output(wait=True)
        return fig

    def train_ResNet(
        self,
        plot_curves: bool = True
    ):
        """
        完整训练流程

        参数:
            num_epochs: 训练轮数
            plot_curves: 是否绘制训练曲线

        返回:
            训练历史记录
        """
        start_time = time.time()
        num_epochs=self.max_epoch

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # 1. 训练一个epoch
            train_loss, train_accuracy = self.train_epoch()

            # 2. 记录训练指标
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_accuracy)


            test_loss, test_accuracy = self.evaluate()
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_accuracy)

            # 5. 记录耗时
            epoch_time = time.time() - epoch_start_time
            self.history['epoch_time'].append(epoch_time)

        self.print_last_epoch(name='ResNet')
        fig = self.board.fig  # 获取当前图形
        fig.suptitle(f'batch_sizes={self.model.batch_size}_lr={self.model.lr}')
        display.display(fig)  # 显示图形
        display.clear_output(wait=True)
        return fig

    def train_PretrainedResNet(
        self,
        plot_curves: bool = True
    ):
        """
        完整训练流程

        参数:
            num_epochs: 训练轮数
            plot_curves: 是否绘制训练曲线

        返回:
            训练历史记录
        """
        start_time = time.time()
        num_epochs=self.max_epoch

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # 1. 训练一个epoch
            train_loss, train_accuracy = self.train_epoch()

            # 2. 记录训练指标
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_accuracy)


            test_loss, test_accuracy = self.evaluate()
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_accuracy)

            # 5. 记录耗时
            epoch_time = time.time() - epoch_start_time
            self.history['epoch_time'].append(epoch_time)

        self.print_last_epoch(name='PretrainedResNet')
        fig = self.board.fig  # 获取当前图形
        fig.suptitle(f'batch_sizes={self.model.batch_size}_lr={self.model.lr}')
        display.display(fig)  # 显示图形
        display.clear_output(wait=True)
        return fig

    
# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 切换到该目录
os.chdir(current_dir)

figs = []
output_dir = os.path.join(current_dir,"..","result", "week06_charts")
os.makedirs(output_dir, exist_ok=True)
data = FashionMNISTData()

# lr, batch_size =0.01,128
# model = AlexNet(num_outputs=10, lr=lr,batch_size=batch_size)
# trainer = Trainer(model=model, data=data,max_epoch=10)
# model.apply_init([next(iter(data.get_train_loader(batch_size=batch_size)))[0]],model.init_cnn)
# fig = trainer.train_AlexNet()
# path = os.path.join(output_dir, f"AlexNet_batch_size={batch_size}_lr={lr}.jpg")
# plt.savefig(path, format='jpg', bbox_inches='tight')
# # figs.append(fig)
# # plt.show() 

# arch,lr, batch_size =((1, 16), (1, 32), (2, 64), (2, 128), (2, 128)),0.01,128
# model = VGG(num_outputs=10, lr=lr,batch_size=batch_size,arch=arch)
# model.apply_init([next(iter(data.get_train_loader(batch_size=batch_size)))[0]],model.init_cnn)
# trainer = Trainer(model=model, data=data,max_epoch=10)
# fig = trainer.train_VGG()
# path = os.path.join(output_dir, f"AlexNet_batch_size={batch_size}_lr={lr}.jpg")
# plt.savefig(path, format='jpg', bbox_inches='tight')

# arch,lr, batch_size =((2, 64), (2, 128), (2, 256), (2, 512)),0.01,128
# model = ResNet(num_outputs=10, lr=lr,batch_size=batch_size,arch=arch)
# model.apply_init([next(iter(data.get_train_loader(batch_size=batch_size)))[0]],model.init_cnn)
# data.dataset_Resize((96, 96))
# trainer = Trainer(model=model, data=data,max_epoch=10)
# fig = trainer.train_ResNet()
# path = os.path.join(output_dir, f"AlexNet_batch_size={batch_size}_lr={lr}.jpg")
# plt.savefig(path, format='jpg', bbox_inches='tight')

lr, batch_size =0.001,128
model = PretrainedResNet(num_outputs=10, lr=lr,batch_size=batch_size,freeze_backbone=True)
data.dataset_Resize((96, 96),rgb=True)
trainer = Trainer(model=model, data=data,max_epoch=10)
fig = trainer.train_PretrainedResNet()
path = os.path.join(output_dir, f"freeze_backbone=True_batch_size={batch_size}_lr={lr}.jpg")
plt.savefig(path, format='jpg', bbox_inches='tight')

lr, batch_size =0.001,128
model = PretrainedResNet(num_outputs=10, lr=lr,batch_size=batch_size,freeze_backbone=False)
data.dataset_Resize((96, 96),rgb=True)
trainer = Trainer(model=model, data=data,max_epoch=10)
fig = trainer.train_PretrainedResNet()
path = os.path.join(output_dir, f"freeze_backbone=False_batch_size={batch_size}_lr={lr}.jpg")
plt.savefig(path, format='jpg', bbox_inches='tight')