"""
train.py - 配置驱动的训练入口

用法:
    python train.py                                  # 用 config.yaml
    python train.py --model resnet18 --epochs 20     # CLI 覆盖

依赖: dataset.py (build_data) / model.py (build_model) / config.yaml
标准化输出:
    - best_model.pth      (按 save_best_by 指标保存最佳权重)
    - train_*.csv         (每 epoch 指标)
    - run_*.txt           (超参/种子/版本/最佳 epoch)
    - curves_*.jpg        (loss/acc 曲线)
"""
import torch
import torch.nn.functional as F
import collections
from torch import nn
import matplotlib
import matplotlib.pyplot as plt
import time
import os
import csv
import sys
import platform
import argparse
import yaml
from datetime import datetime
from pathlib import Path

# IPython display 可选（notebook 用动画，脚本环境用静态图）
try:
    from IPython import display
    import matplotlib_inline
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

from dataset import build_data
from model import build_model


# ============================================================
# 可复现性
# ============================================================
def set_seed(seed=42, deterministic=True, benchmark=False):
    """设置随机种子，保障可复现。"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = benchmark


# ============================================================
# ProgressBoard（保留源码，仅在 IPython 环境动画显示）
# ============================================================
class ProgressBoard:
    """The board that plots data points in animation."""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.xscale = xscale
        self.yscale = yscale
        self.ls = ls
        self.colors = colors
        self.fig = fig
        self.axes = axes
        self.figsize = figsize
        self.display = display

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
        if HAS_IPYTHON:
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
        if HAS_IPYTHON:
            display.display(self.fig)
            display.clear_output(wait=True)


# ============================================================
# Trainer
# ============================================================
class Trainer:
    """通用训练器：自管 device/loss_fn/optimizer/batch_size，适配无这些属性的模型。"""

    def __init__(self, model, data, cfg, model_name='model', verbose=True):
        """
        参数:
            model:       模型实例（无需自带 device/loss_fn/optimizer）
            data:        数据管理器（build_data 返回，有 get_train/test_loader）
            cfg:         完整配置字典
            model_name:  模型名（用于日志文件命名）
            verbose:     是否打印每轮指标
        """
        self.model = model
        self.data = data
        self.cfg = cfg
        self.model_name = model_name
        self.verbose = verbose

        # ---- 设备 ----
        device_str = cfg.get('train', {}).get('device', 'cuda')
        if device_str == 'cuda' and not torch.cuda.is_available():
            device_str = 'cpu'
        self.device = torch.device(device_str)
        self.model.to(self.device)

        # ---- 训练参数 ----
        train_cfg = cfg.get('train', {})
        data_cfg = cfg.get('data', {})
        self.batch_size = data_cfg.get('batch_size', 128)
        self.lr = train_cfg.get('lr', train_cfg.get('lr', 0.001))
        self.max_epoch = train_cfg.get('max_epoch', 10)

        # ---- 损失函数（Trainer 自管，模型不再提供）----
        self.loss_fn = F.cross_entropy

        # ---- 优化器（Trainer 自管，按 train.optimizer 配置）----
        opt_type = train_cfg.get('optimizer', 'sgd')
        weight_decay = train_cfg.get('weight_decay', 0.0)
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        if opt_type == 'sgd':
            self.optimizer = torch.optim.SGD(
                params, lr=self.lr,
                momentum=train_cfg.get('momentum', 0.9),
                weight_decay=weight_decay)
        elif opt_type == 'adam':
            self.optimizer = torch.optim.Adam(
                params, lr=self.lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"未知 optimizer: {opt_type}")

        # ---- DataLoader ----
        self.train_loader = self.data.get_train_loader()
        self.test_loader = self.data.get_test_loader()

        # ---- Lazy 层初始化：dummy forward 触发 shape 推断 ----
        self._init_lazy_layers()

        # ---- 历史记录 ----
        self.history = {
            'train_loss': [], 'train_acc': [],
            'test_loss': [], 'test_acc': [],
            'lr': [], 'epoch_time': []
        }

        # ---- 最佳模型追踪 ----
        output_cfg = cfg.get('output', {})
        self.save_best_by = output_cfg.get('save_best_by', 'test_loss')
        if self.save_best_by == 'test_loss':
            self.best_val = float('inf')
        else:
            self.best_val = -float('inf')
        self.best_epoch = -1

        # ---- 早停 ----
        es_cfg = train_cfg.get('early_stopping', {})
        self.early_stopping = es_cfg.get('enabled', False)
        self.patience = es_cfg.get('patience', 5)
        self.es_metric = es_cfg.get('metric', 'test_loss')
        self.es_mode = es_cfg.get('mode', 'min')
        self.es_counter = 0
        self.es_best = float('inf') if self.es_mode == 'min' else -float('inf')

        # ---- 输出路径 ----
        out_root = Path(output_cfg.get('root', './result'))
        self.ckpt_dir = out_root / output_cfg.get('checkpoint_dir', 'checkpoints')
        self.log_dir = out_root / output_cfg.get('log_dir', 'logs')
        self.chart_dir = out_root / output_cfg.get('chart_dir', 'charts')
        for d in (self.ckpt_dir, self.log_dir, self.chart_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.ckpt_dir / output_cfg.get(
            'best_model_name', 'best_model.pth')

        # ---- 日志开关 ----
        log_cfg = cfg.get('logging', {})
        self.csv_log = log_cfg.get('csv_log', True)
        self.run_meta = log_cfg.get('run_meta', True)
        self.plot_curves = log_cfg.get('plot_curves', True)

        # ---- 绘图板 ----
        self.board = ProgressBoard(display=HAS_IPYTHON)

        # ---- 训练状态 ----
        self.current_epoch = 0
        self.train_batch_idx = 0
        self.is_trained = False

    # ----------------------------------------------------------
    # Lazy 层 shape 推断（AlexNet 等用 LazyConv2d/LazyLinear）
    # ----------------------------------------------------------
    def _init_lazy_layers(self):
        """取一个 batch 做 forward，触发 Lazy 层参数实例化。"""
        try:
            X, _ = next(iter(self.train_loader))
            X = X.to(self.device)
            self.model.eval()
            with torch.no_grad():
                _ = self.model(X)
        except StopIteration:
            pass

    # ----------------------------------------------------------
    # 绘图（保留源码逻辑）
    # ----------------------------------------------------------
    def plot(self, key, value, train):
        """Plot a point in animation."""
        self.board.xlabel = 'epoch'
        plot_train_per_epoch = 2
        plot_valid_per_epoch = 1
        num_train_batches = len(self.train_loader)
        num_test_batches = len(self.test_loader)
        if train:
            x = self.train_batch_idx / \
                num_train_batches + self.current_epoch - 1
            n = num_train_batches / plot_train_per_epoch
        else:
            x = self.current_epoch
            n = num_test_batches / plot_valid_per_epoch
        self.board.draw(x, value.cpu().detach().numpy(),
                        ('train_' if train else 'test_') + key,
                        every_n=int(n))

    # ----------------------------------------------------------
    # 指标（保留源码）
    # ----------------------------------------------------------
    def accuracy(self, Y_hat, Y, averaged=True):
        """Compute the number of correct predictions."""
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        preds = Y_hat.argmax(axis=1).type(Y.dtype)
        compare = (preds == Y.reshape(-1)).type(torch.float32)
        return compare.mean() if averaged else compare

    # ----------------------------------------------------------
    # 单 epoch 训练（保留源码逻辑，改用 self.device）
    # ----------------------------------------------------------
    def train_epoch(self):
        """训练一个 epoch，返回 (avg_loss, accuracy)。"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        self.train_batch_idx = 0

        for X, y in self.train_loader:
            X, y = X.to(self.device), y.to(self.device)

            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)

            self.plot('loss', loss, train=True)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.train_batch_idx += 1

            total_loss += loss.item()
            _, predicted = torch.max(y_pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        acc = correct / total
        return avg_loss, acc

    # ----------------------------------------------------------
    # 评估（保留源码逻辑，改用 self.device）
    # ----------------------------------------------------------
    def evaluate(self):
        """评估模型，返回 (avg_loss, accuracy)。"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                y_pred = self.model(X)
                loss = self.loss_fn(y_pred, y)

                self.plot('loss', loss, train=False)
                self.plot('acc', self.accuracy(y_pred, y), train=False)

                total_loss += loss.item()
                _, predicted = torch.max(y_pred, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        avg_loss = total_loss / len(self.test_loader)
        acc = correct / total
        return avg_loss, acc

    # ----------------------------------------------------------
    # 打印（保留源码）
    # ----------------------------------------------------------
    def print_last_epoch(self, name):
        """打印最后一个 epoch 的训练结果。"""
        history = self.history
        if not history['train_loss']:
            print("暂无训练数据！")
            return
        last_idx = -1
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
        """打印所有轮次的训练结果。"""
        history = self.history
        if not history['train_loss']:
            print("暂无训练数据！")
            return
        num_epochs = len(history['train_loss'])
        print("=" * 100)
        print(f"{name:^100}")
        print("=" * 100)
        print(f"{'Epoch':<8} {'Train Loss':<15} {'Train Acc':<15} "
              f"{'Test Loss':<15} {'Test Acc':<15} {'Time(s)':<12}")
        print("-" * 100)
        for i in range(num_epochs):
            print(f"{i+1:<8} "
                  f"{history['train_loss'][i]:<15.4f} "
                  f"{history['train_acc'][i]:<15.4f} "
                  f"{history['test_loss'][i]:<15.4f} "
                  f"{history['test_acc'][i]:<15.4f} "
                  f"{history['epoch_time'][i]:<12.2f}")
        print("=" * 100)

    # ----------------------------------------------------------
    # 统一训练入口（合并原 5 个 train_XxxNet）
    # ----------------------------------------------------------
    def train(self, name=None):
        """
        完整训练流程（替代原 train_AlexNet/VGG/GoogleNet/ResNet/PretrainedResNet）。
        参数 name 仅用于打印标题与日志文件命名。
        """
        name = name or self.model_name
        start_time = time.time()

        for epoch in range(1, self.max_epoch + 1):
            if self.verbose:
                print(f"Epoch :{epoch},start")
            self.current_epoch = epoch
            epoch_start = time.time()

            # 1. 训练 + 评估
            train_loss, train_acc = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            test_loss, test_acc = self.evaluate()
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)

            # 2. 记录 lr 与耗时
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            epoch_time = time.time() - epoch_start
            self.history['epoch_time'].append(epoch_time)

            # 3. 保存最佳模型
            if self._is_best(test_loss, test_acc):
                self._save_checkpoint(epoch, test_loss, test_acc)

            # 4. 打印本轮
            if self.verbose:
                self._print_epoch(epoch, train_loss, train_acc,
                                  test_loss, test_acc, epoch_time)

            # 5. 早停
            if self.early_stopping and self._check_early_stop(test_loss, test_acc):
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        # ---- 训练结束 ----
        total_time = time.time() - start_time
        self.is_trained = True

        if self.verbose:
            self.print_last_epoch(name=name)
            print(f"Total time: {total_time:.1f}s | "
                  f"Best epoch: {self.best_epoch} | "
                  f"Best {self.save_best_by}: {self.best_val:.4f}")

        if self.csv_log:
            self._save_csv(name)
        if self.run_meta:
            self._save_run_meta(name, total_time)
        if self.plot_curves:
            self._save_curves(name)

        return self.history

    # ----------------------------------------------------------
    # 最佳模型判断 + 保存
    # ----------------------------------------------------------
    def _is_best(self, test_loss, test_acc):
        if self.save_best_by == 'test_loss':
            if test_loss < self.best_val:
                self.best_val = test_loss
                self.best_epoch = self.current_epoch
                return True
        else:  # test_acc
            if test_acc > self.best_val:
                self.best_val = test_acc
                self.best_epoch = self.current_epoch
                return True
        return False

    def _save_checkpoint(self, epoch, test_loss, test_acc):
        torch.save(self.model.state_dict(), self.best_model_path)
        if self.verbose:
            metric_val = test_loss if self.save_best_by == 'test_loss' else test_acc
            print(f"  -> New best ({self.save_best_by}={metric_val:.4f}) "
                  f"saved to {self.best_model_path.name}")

    # ----------------------------------------------------------
    # 早停
    # ----------------------------------------------------------
    def _check_early_stop(self, test_loss, test_acc):
        val = test_loss if self.es_metric == 'test_loss' else test_acc
        improved = (val < self.es_best) if self.es_mode == 'min' else (val > self.es_best)
        if improved:
            self.es_best = val
            self.es_counter = 0
        else:
            self.es_counter += 1
        return self.es_counter >= self.patience

    # ----------------------------------------------------------
    # 打印单行
    # ----------------------------------------------------------
    def _print_epoch(self, epoch, tl, ta, vl, va, t):
        print(f"Epoch {epoch}/{self.max_epoch} | "
              f"train_loss={tl:.4f} train_acc={ta:.4f} | "
              f"test_loss={vl:.4f} test_acc={va:.4f} | "
              f"time={t:.1f}s")

    # ----------------------------------------------------------
    # CSV 日志
    # ----------------------------------------------------------
    def _save_csv(self, name):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = self.log_dir / f"train_{name}_{ts}.csv"
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc',
                             'test_loss', 'test_acc', 'lr', 'epoch_time'])
            for i in range(len(self.history['train_loss'])):
                writer.writerow([
                    i + 1,
                    self.history['train_loss'][i],
                    self.history['train_acc'][i],
                    self.history['test_loss'][i],
                    self.history['test_acc'][i],
                    self.history['lr'][i],
                    self.history['epoch_time'][i],
                ])

    # ----------------------------------------------------------
    # Run 元信息
    # ----------------------------------------------------------
    def _save_run_meta(self, name, total_time):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = self.log_dir / f"run_{name}_{ts}.txt"
        data_cfg = self.cfg.get('data', {})
        train_cfg = self.cfg.get('train', {})
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"Model: {name}\n")
            f.write(f"Dataset: {data_cfg.get('name', 'unknown')}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"max_epoch: {self.max_epoch} "
                    f"(actually trained {len(self.history['train_loss'])})\n")
            f.write(f"batch_size: {self.batch_size}\n")
            f.write(f"lr: {self.lr}\n")
            f.write(f"optimizer: {type(self.optimizer).__name__}\n")
            f.write(f"weight_decay: {train_cfg.get('weight_decay', 0.0)}\n")
            f.write(f"seed: {self.cfg.get('seed', 42)}\n")
            f.write(f"deterministic: {self.cfg.get('deterministic', True)}\n")
            f.write(f"torch_version: {torch.__version__}\n")
            f.write(f"python_version: {platform.python_version()}\n")
            f.write(f"best_epoch: {self.best_epoch}\n")
            f.write(f"best_{self.save_best_by}: {self.best_val:.4f}\n")
            f.write(f"total_time: {total_time:.1f}s\n")

    # ----------------------------------------------------------
    # 训练曲线（脚本环境用 Agg 后端存静态图）
    # ----------------------------------------------------------
    def _save_curves(self, name):
        if not HAS_IPYTHON:
            matplotlib.use('Agg')
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, len(self.history['train_loss']) + 1)

        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='train_loss')
        axes[0].plot(epochs, self.history['test_loss'], 'r--', label='test_loss')
        axes[0].set_xlabel('epoch')
        axes[0].set_ylabel('loss')
        axes[0].legend()
        axes[0].set_title('Loss')

        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='train_acc')
        axes[1].plot(epochs, self.history['test_acc'], 'r--', label='test_acc')
        axes[1].set_xlabel('epoch')
        axes[1].set_ylabel('accuracy')
        axes[1].legend()
        axes[1].set_title('Accuracy')

        fig.suptitle(f"{name} | batch_size={self.batch_size} lr={self.lr}")
        plt.tight_layout()
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = self.chart_dir / f"curves_{name}_{ts}.jpg"
        plt.savefig(path, dpi=100)
        plt.close(fig)


# ============================================================
# 配置加载 + CLI
# ============================================================
def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='训练入口')
    parser.add_argument('--config', default='config.yaml',
                        help='配置文件路径（默认 config.yaml）')
    parser.add_argument('--model', help='覆盖 cfg.model.name')
    parser.add_argument('--data', help='覆盖 cfg.data.name')
    parser.add_argument('--epochs', type=int, help='覆盖 cfg.train.max_epoch')
    parser.add_argument('--lr', type=float, help='覆盖 cfg.model.lr')
    parser.add_argument('--batch-size', type=int,
                        help='覆盖 cfg.data.batch_size')
    parser.add_argument('--device', help='覆盖 cfg.model.device (cpu/cuda)')
    parser.add_argument('--no-freeze', action='store_true',
                        help='迁移学习时不冻结骨干（仅 resnet18/vgg16 + pretrained 时生效）')
    return parser.parse_args()


def main():
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 切换到该目录
    os.chdir(current_dir)

    args = parse_args()
    cfg = load_config(args.config)

    # CLI 覆盖
    if args.model:
        cfg['model']['name'] = args.model
    if args.data:
        cfg['data']['name'] = args.data
    if args.epochs:
        cfg['train']['max_epoch'] = args.epochs
    if args.lr:
        cfg['model']['lr'] = args.lr
    if args.batch_size:
        cfg['data']['batch_size'] = args.batch_size
    if args.device:
        cfg['model']['device'] = args.device

    # 可复现
    set_seed(cfg.get('seed', 42),
             cfg.get('deterministic', True),
             cfg.get('benchmark', False))

    # 构建数据与模型
    data = build_data(cfg)
    model = build_model(cfg)           # 新签名：收整个 config

    model_name = cfg['model']['name']

    # 训练
    trainer = Trainer(model, data, cfg,
                      model_name=model_name,
                      verbose=cfg.get('logging', {}).get('verbose', True))
    # ===== 调试打印 =====
    print("=== 可训练参数 ===")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print("  ", name)

    print("=== conv1 权重统计 ===")
    print("  mean:", model.model.conv1.weight.mean().item())
    print("  std :", model.model.conv1.weight.std().item())

    print("=== fc 输出维度 ===", model.model.fc.out_features)
    print("=== 优化器 ===", trainer.optimizer)
    print("=== 模型模式 ===", model.training)
    print("=== 设备 ===", next(model.parameters()).device)

    # 取一个 batch 看数据
    images, labels = next(iter(data.get_train_loader()))
    print("=== 数据形状 ===", images.shape)
    print("=== 数据范围 ===", images.min().item(), images.max().item())
    print("=== 标签样例 ===", labels[:10])
    trainer.train(name=model_name)


if __name__ == '__main__':
    main()
