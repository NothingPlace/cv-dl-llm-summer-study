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

def init_cnn(module):  #@save
    """Initialize weights for CNNs."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)

class LeNet(nn.Module):
    """LeNet 模型"""
    def __init__(self, num_outputs,batch_size=256,lr=0.1, device='cpu'):
        super().__init__()
        self.batch_size=batch_size
        self.lr=lr

        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.Sigmoid(),
            nn.LazyLinear(84), nn.Sigmoid(),
            nn.LazyLinear(num_outputs))
        # 损失函数
        self.loss_fn = F.cross_entropy
        # 优化器
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        # 移动到设备
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, X):

        return self.net(X)
    
    def apply_init(self, inputs, init=None):
        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)

class MLPRegression(nn.Module):
    """MLP 回归模型"""
    def __init__(self, num_outputs,batch_size=256,lr=0.1, device='cpu',
                 num_hiddens_1=256,num_hiddens_2=256,dropout_1=0.5,dropout_2=0.5):
        super().__init__()
        self.batch_size=batch_size
        self.lr=lr
        self.num_hiddens_1=num_hiddens_1
        self.num_hiddens_2=num_hiddens_2
        self.dropout_1=dropout_1
        self.dropout_2=dropout_2

        self.net = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(num_hiddens_1), nn.ReLU(),
            nn.Dropout(dropout_1), nn.LazyLinear(num_hiddens_2), nn.ReLU(),
            nn.Dropout(dropout_2), nn.LazyLinear(num_outputs))
        # 损失函数
        self.loss_fn = F.cross_entropy
        # 优化器
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        # 移动到设备
        self.device = torch.device(device)
        self.to(self.device)
    def forward(self, X):
        # 将图像展平为一维向量
        X = X.view(X.shape[0], -1)  # (batch_size, 784)
        return self.net(X)

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
        model: LeNet,
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
        self.board.draw(x, value.detach().numpy(),
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

    def train_LeNet(
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

        fig = self.board.fig  # 获取当前图形
        fig.suptitle(f'batch_sizes={self.model.batch_size}_lr={self.model.lr}')
        display.display(fig)  # 显示图形
        display.clear_output(wait=True)
        return fig

    def train_MLP(
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

            fig = self.board.fig  # 获取当前图形
            fig.suptitle(f'num_hiddens_1={self.model.num_hiddens_1}_num_hiddens_2={self.model.num_hiddens_2}_dropout_1={self.model.dropout_1}_dropout_2={self.model.dropout_2}')
            display.display(fig)  # 显示图形
            display.clear_output(wait=True)
            return fig


    
# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 切换到该目录
os.chdir(current_dir)

figs = []
output_dir = os.path.join(current_dir,"..","result", "week05_line_charts")
os.makedirs(output_dir, exist_ok=True)
data = FashionMNISTData()

lr, batch_size =0.1,128
model = LeNet(num_outputs=10, lr=lr,batch_size=batch_size)
trainer = Trainer(model=model, data=data,max_epoch=10)
model.apply_init([next(iter(data.get_train_loader(batch_size=batch_size)))[0]], init_cnn)
fig = trainer.train_LeNet()
path = os.path.join(output_dir, f"LeNet_batch_size={batch_size}_lr={lr}.jpg")
plt.savefig(path, format='jpg', bbox_inches='tight')
# figs.append(fig)
# plt.show() 

X, y = next(iter(data.get_test_loader()))
preds = model(X).argmax(axis=1)
labels = [a+'\n'+b for a, b in zip(
    data.text_labels(y), data.text_labels(preds))]
image=data.visualize([X, y], labels=labels)
path = os.path.join(output_dir, f"predict.jpg")
plt.savefig(path, format='jpg', bbox_inches='tight')

wrong = preds.type(y.dtype) != y
X, y, preds = X[wrong], y[wrong], preds[wrong]
labels = [a+'\n'+b for a, b in zip(
    data.text_labels(y), data.text_labels(preds))]
image=data.visualize([X, y], labels=labels)
path = os.path.join(output_dir, f"wrong.jpg")
plt.savefig(path, format='jpg', bbox_inches='tight')

lr, batch_size,num_hiddens_1,num_hiddens_2,dropout_1,dropout_2=0.1,128,256,256,0.5,0.5
model = MLPRegression(num_outputs=10, lr=lr,batch_size=batch_size,
                      num_hiddens_1=num_hiddens_1,num_hiddens_2=num_hiddens_2,
                      dropout_1=dropout_1,dropout_2=dropout_2)
trainer = Trainer(model=model, data=data,max_epoch=10)
fig = trainer.train_MLP()
path = os.path.join(output_dir, f"MLP_num_hiddens_1={num_hiddens_1}_num_hiddens_2={num_hiddens_2}_dropout_1={dropout_1}_dropout_2={dropout_2}.jpg")
plt.savefig(path, format='jpg', bbox_inches='tight')