"""
test.py - 配置驱动的测试/推理入口

用法:
    python test.py                                    # 用 config.yaml，加载 best_model.pth
    python test.py --model resnet18 --ckpt best.pth   # CLI 覆盖
    python test.py --predict-single 0                 # 对测试集第 0 张做单图预测

依赖: dataset.py (build_data) / model.py (build_model) / config.yaml
输出:
    - 控制台打印 test loss / accuracy
    - charts/confusion_matrix_<model>.jpg  (需 scikit-learn)
"""
import torch
import torch.nn.functional as F
from torch import nn
import argparse
import csv
import yaml
import matplotlib
matplotlib.use('Agg')   # 脚本环境，不弹窗
import matplotlib.pyplot as plt
# Windows 中文类别名显示（字体不存在时 matplotlib 自动回退，不报错）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from pathlib import Path
from datetime import datetime

from dataset import build_data
from model import build_model


# ============================================================
# 配置加载 + CLI
# ============================================================
def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='week07 测试/推理入口')
    parser.add_argument('--config', default='config.yaml',
                        help='配置文件路径（默认 config.yaml）')
    parser.add_argument('--model', help='覆盖 cfg.model.name')
    parser.add_argument('--data', help='覆盖 cfg.data.name')
    parser.add_argument('--ckpt', help='覆盖 cfg.test.load_ckpt（文件名或绝对路径）')
    parser.add_argument('--batch-size', type=int, help='覆盖 cfg.data.batch_size')
    parser.add_argument('--device', help='覆盖 cfg.model.device (cpu/cuda)')
    parser.add_argument('--predict-single', type=int, default=None,
                        help='对测试集指定 index 做单图预测并可视化，不评估全量')
    parser.add_argument('--num-errors', type=int, default=None,
                        help='错误样例展示数量（覆盖 test.error_samples_num，设 0 关闭）')
    parser.add_argument('--num-preds', type=int, default=None,
                        help='预测展示数量（覆盖 test.prediction_samples_num，设 0 关闭）')
    return parser.parse_args()


def get_device(cfg, override=None):
    """设备选择：CLI > config；cuda 不可用时自动回退 cpu。"""
    device_str = override or cfg.get('train', {}).get('device', 'cuda')
    if device_str == 'cuda' and not torch.cuda.is_available():
        print('[!] CUDA 不可用，回退到 CPU')
        device_str = 'cpu'
    return torch.device(device_str)


# ============================================================
# 模型构建 + Lazy 初始化 + 权重加载
# ============================================================
def load_model(cfg, ckpt_path, device):
    """构建模型 → dummy forward 触发 Lazy 层实例化 → 加载权重。"""
    model = build_model(cfg)
    model.to(device)
    model.eval()

    # Lazy 层（AlexNet 的 LazyConv2d/LazyLinear）需先 forward 才能确定 shape，
    # 否则 load_state_dict 找不到对应参数键。
    _init_lazy_layers(model, cfg, device)

    if ckpt_path and Path(ckpt_path).exists():
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[!] {len(missing)} 个键缺失（前 5 个）: {missing[:5]}")
        if unexpected:
            print(f"[!] {len(unexpected)} 个多余键（前 5 个）: {unexpected[:5]}")
        print(f"[OK] 已加载权重: {ckpt_path}")
    elif ckpt_path:
        print(f"[!] 权重文件不存在: {ckpt_path}，使用随机初始化权重评估")
    else:
        print("[!] 未指定 ckpt，使用随机初始化权重评估")
    return model

def get_model_input_channels(model):
    """返回模型第一层卷积的输入通道数；找不到则返回 None。"""
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            # LazyConv2d 未初始化时 in_channels 可能是 0 或 None
            if getattr(module, "in_channels", None):
                return module.in_channels
            return None
        if isinstance(module, nn.Linear):
            # 没有卷积，遇到 Linear 就说明不是图像模型
            return None
    return None

def _init_lazy_layers(model, cfg, device):
    """取一个测试 batch 做 dummy forward，触发 Lazy 层 shape 推断。"""
    data_cfg = cfg.get('data', {})
    resize = tuple(data_cfg.get('resize', (224, 224)))
    channels = get_model_input_channels(model)
    batch_size = data_cfg.get('batch_size', 4)
    dummy = torch.randn(batch_size, channels, *resize, device=device)
    with torch.no_grad():
        _ = model(dummy)


def resolve_ckpt_path(cfg, override):
    """解析权重路径：CLI 优先；配置里为相对 output.checkpoint_dir 的文件名。"""
    if override:
        p = Path(override)
        return p if p.is_absolute() else Path.cwd() / p
    test_cfg = cfg.get('test', {})
    ckpt_name = test_cfg.get('load_ckpt', 'best_model.pth')
    if ckpt_name is None:
        return None
    out_cfg = cfg.get('output', {})
    root = Path(out_cfg.get('root', '../result/week07'))
    ckpt_dir = root / out_cfg.get('checkpoint_dir', 'checkpoints')
    return ckpt_dir / ckpt_name


# ============================================================
# 评估
# ============================================================
def evaluate(model, loader, device, loss_fn=F.cross_entropy):
    """全量评估，返回 (avg_loss, accuracy, y_true, y_pred, indices)。

    要求 loader shuffle=False，indices 才能对齐 data.test_dataset 的全局索引，
    供错误样例回溯原图使用。
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true, y_pred, indices = [], [], []
    global_idx = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()
            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            bs = y.size(0)
            indices.extend(range(global_idx, global_idx + bs))
            global_idx += bs

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy, y_true, y_pred, indices


# ============================================================
# 混淆矩阵
# ============================================================
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """绘制混淆矩阵，保存到 save_path。需 scikit-learn。"""
    try:
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    except ImportError:
        print("[!] 未安装 scikit-learn，跳过混淆矩阵（pip install scikit-learn）")
        return None

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45, colorbar=False)
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] 混淆矩阵已保存: {save_path}")

    # 各类别准确率
    print("\n各类别召回率（对角线召回）:")
    for i, name in enumerate(class_names):
        if cm[i].sum() > 0:
            recall = cm[i, i] / cm[i].sum()
            print(f"  {name:<12} {recall:.3f}")
    return cm

def plot_confusion_matrix_top_n(y_true, y_pred, class_names, save_path, top_n=20):
    """绘制混淆矩阵（只画错误最多的 Top-N 类别），保存到 save_path。需 scikit-learn。"""
    try:
        import numpy as np
        from sklearn.metrics import confusion_matrix
    except ImportError:
        print("[!] 未安装 scikit-learn，跳过混淆矩阵（pip install scikit-learn）")
        return None

    # 1. 完整混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 2. 计算每个类别的错误数（该行总数 - 对角线）
    errors = cm.sum(axis=1) - np.diag(cm)

    # 3. 取错误最多的 top_n 个类别索引，排序后画出来更整齐
    top_idx = np.argsort(errors)[::-1][:top_n]
    top_idx = sorted(top_idx.tolist())

    # 4. 提取子矩阵和对应标签
    cm_sub = cm[np.ix_(top_idx, top_idx)]
    labels = [class_names[i] for i in top_idx]

    # 5. 画图
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_sub, cmap='Blues')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 设置刻度与标签
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    # 在每个格子里标数字
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm_sub[i, j]
            if val > 0:
                ax.text(j, i, str(val), ha='center', va='center',
                        fontsize=7,
                        color='white' if val > cm_sub.max() / 2 else 'black')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix (Top {top_n} most confused classes)')
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] 混淆矩阵（Top {top_n}）已保存: {save_path}")

    # 各类别召回率（只打印 Top-N 里的类别）
    print(f"\nTop {top_n} 易混淆类别的召回率:")
    for i in top_idx:
        if cm[i].sum() > 0:
            recall = cm[i, i] / cm[i].sum()
            print(f"  {class_names[i]:<25} 召回率={recall:.3f}  "
                  f"错误数={errors[i]}")

    return cm


# ============================================================
# 类别级准确率（CSV 数据文件 + 条形图）
# ============================================================
def export_per_class_metrics(y_true, y_pred, class_names,
                             csv_path=None, fig_path=None):
    """计算每个类别的 precision / recall(类别准确率) / f1 / support。

    - CSV: 全量类别明细，按 recall 升序（最差类别在最前），方便排查
    - 条形图: 每类 recall，颜色分级（红<0.5 / 橙<0.8 / 绿>=0.8），
              测试集中缺失的类别(support=0)灰色排末尾
    需 scikit-learn。
    """
    try:
        import numpy as np
        from sklearn.metrics import precision_recall_fscore_support
    except ImportError:
        print("[!] 未安装 scikit-learn，跳过类别级指标（pip install scikit-learn）")
        return

    labels = list(range(len(class_names)))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)

    # ---- 控制台：宏平均 / 加权平均 + 最差 10 类 ----
    n_present = int((support > 0).sum())
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    print(f"\n类别级指标（测试集覆盖 {n_present}/{len(class_names)} 类）:")
    print(f"  macro:    precision={macro_p:.4f} recall={macro_r:.4f} f1={macro_f1:.4f}")
    print(f"  weighted: precision={weighted_p:.4f} recall={weighted_r:.4f} f1={weighted_f1:.4f}")

    present_idx = [i for i in labels if support[i] > 0]
    worst = sorted(present_idx, key=lambda i: r[i])[:10]
    print("  recall 最低的 10 个类别:")
    for i in worst:
        print(f"    {class_names[i]:<25} recall={r[i]:.3f} support={int(support[i])}")
    missing = [class_names[i] for i in labels if support[i] == 0]
    if missing:
        print(f"  [!] {len(missing)} 个类别在测试集中无样本（CSV 中指标留空）")

    # ---- CSV 数据文件（recall 升序，support=0 的类别排末尾）----
    if csv_path is not None:
        rows = []
        for i in present_idx:
            rows.append((i, class_names[i], p[i], r[i], f1[i], int(support[i])))
        rows.sort(key=lambda row: row[3])                       # recall 升序
        rows += [(i, class_names[i], None, None, None, 0)
                 for i in labels if support[i] == 0]

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['class_index', 'class_name',
                             'precision', 'recall', 'f1', 'support'])
            for idx, name, pp, rr, ff, ss in rows:
                writer.writerow([
                    idx, name,
                    '' if pp is None else f'{pp:.4f}',
                    '' if rr is None else f'{rr:.4f}',
                    '' if ff is None else f'{ff:.4f}',
                    ss,
                ])
        print(f"[OK] 类别级指标 CSV 已保存: {csv_path}")

    # ---- 条形图（横向，按 recall 升序，适配 102 类这种多类别场景）----
    if fig_path is not None:
        order = sorted(present_idx, key=lambda i: r[i])
        order += [i for i in labels if support[i] == 0]
        names = [class_names[i] for i in order]
        recalls_sorted = [r[i] if support[i] > 0 else 0.0 for i in order]

        def bar_color(v, has_support):
            if not has_support:
                return '#cccccc'
            if v < 0.5:
                return '#d62728'      # 红
            if v < 0.8:
                return '#ff7f0e'      # 橙
            return '#2ca02c'          # 绿

        colors = [bar_color(r[i], support[i] > 0) for i in order]
        n = len(order)
        fig_h = max(6.0, n * 0.28)
        fig, ax = plt.subplots(figsize=(10, fig_h))
        y = np.arange(n)
        ax.barh(y, recalls_sorted, color=colors, height=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=6 if n > 50 else 8)
        ax.invert_yaxis()                 # recall 最低的在最上方
        ax.set_xlim(0, 1)
        ax.set_xlabel('Recall (per-class accuracy)')
        ax.set_title(f'Per-class Recall | {n} classes | '
                     f'macro={macro_r:.3f} weighted={weighted_r:.3f}')
        ax.axvline(macro_r, color='blue', linestyle='--', linewidth=1,
                   label=f'macro recall={macro_r:.3f}')
        ax.legend(loc='lower right')
        # 在条形末端标数值（类别太多时省略，避免重叠）
        if n <= 50:
            for yi, (i, v) in enumerate(zip(order, recalls_sorted)):
                if support[i] > 0:
                    ax.text(v + 0.01, yi, f'{v:.2f}', va='center', fontsize=6)
        plt.tight_layout()
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[OK] 类别级准确率图已保存: {fig_path}")


# ============================================================
# 单图预测
# ============================================================
def predict_single(model, data, index, device, save_path=None):
    """对测试集指定 index 做预测并可视化。"""
    dataset = data.test_dataset
    img, label = dataset[index]
    model.eval()
    with torch.no_grad():
        logits = model(img.unsqueeze(0).to(device))
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()
        conf = probs[0, pred].item()

    class_names = getattr(data, 'class_names', None) or [str(i) for i in range(logits.shape[-1])]
    true_name = class_names[label]
    pred_name = class_names[pred]
    print(f"\n单图预测 (index={index}):")
    print(f"  真实标签: {true_name}")
    print(f"  预测标签: {pred_name}  (置信度 {conf:.3f})")
    print(f"  结果: {'✓ 正确' if pred == label else '✗ 错误'}")

    # 可视化（反归一化后显示）
    fig, ax = plt.subplots(figsize=(3, 3))
    disp_img = img.permute(1, 2, 0).cpu().numpy()
    disp_img = (disp_img * 0.5 + 0.5).clip(0, 1)   # 逆 Normalize(0.5,0.5)
    if disp_img.shape[2] == 1:
        disp_img = disp_img.squeeze(-1)
    ax.imshow(disp_img, cmap='gray' if disp_img.ndim == 2 else None)
    ax.set_title(f"true: {true_name}\npred: {pred_name} ({conf:.2f})",
                 color='green' if pred == label else 'red')
    ax.axis('off')
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"[OK] 预测可视化已保存: {save_path}")
    plt.close(fig)


# ============================================================
# 错误样例 / 预测展示（网格图：图片 + 真实结果 + 预测结果）
# ============================================================
def _denormalize(img, data_cfg):
    """按配置的 mean/std 做逆归一化，返回 clip 到 [0,1] 的 HWC numpy 图。"""
    import numpy as np
    mean = data_cfg.get('mean', [0.5])
    std = data_cfg.get('std', [0.5])
    rgb_expand = data_cfg.get('rgb_expand', False)
    # rgb 且只给了单通道统计量时，广播到 3 通道（与 dataset.build_transform 一致）
    if rgb_expand and len(mean) == 1:
        mean = mean * 3
    if rgb_expand and len(std) == 1:
        std = std * 3
    mean = torch.tensor(mean, dtype=img.dtype).view(-1, 1, 1)
    std = torch.tensor(std, dtype=img.dtype).view(-1, 1, 1)
    img = img.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    if img.shape[2] == 1:
        img = img.squeeze(-1)
    return img


def plot_prediction_grid(dataset, sample_indices, y_true, y_pred,
                         class_names, save_path, data_cfg,
                         title='', ncols=4, only_errors=False):
    """生成包含 图片/真实结果/预测结果 的网格图。

    参数:
        dataset:        data.test_dataset（按全局索引取 (img, label)）
        sample_indices: 要展示的全局索引列表（错误样例或任意样本）
        y_true/y_pred:  evaluate 收集的预测结果（按全局索引对齐）
        class_names:    类别名列表
        only_errors:    True 时只展示预测错误的样本（标题红色标识）
    """
    if len(sample_indices) == 0:
        print("[!] 没有可展示的样例" + ("（无错误样本）" if only_errors else ""))
        return

    n = len(sample_indices)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.2, nrows * 3.8))
    axes = np_ravel(axes, nrows, ncols)

    for ax, idx in zip(axes, sample_indices):
        img, label = dataset[idx]
        true_id, pred_id = y_true[idx], y_pred[idx]
        true_name = class_names[true_id]
        pred_name = class_names[pred_id]
        is_correct = (true_id == pred_id)

        disp_img = _denormalize(img, data_cfg)
        ax.imshow(disp_img, cmap='gray' if disp_img.ndim == 2 else None)
        color = 'green' if is_correct else 'red'
        ax.set_title(f"true: {true_name}\npred: {pred_name}",
                     fontsize=8, color=color)
        ax.axis('off')

    # 隐藏多余子图
    for ax in axes[len(sample_indices):]:
        ax.axis('off')

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] {title or '样例展示'}已保存: {save_path}（{n} 张）")


def np_ravel(axes, nrows, ncols):
    """兼容 1 行/多行情况下 axes 的扁平化。"""
    import numpy as np
    if nrows == 1 and ncols == 1:
        return [axes]
    return np.array(axes).ravel()


# ============================================================
# 主入口
# ============================================================
import os
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
    if args.batch_size:
        cfg['data']['batch_size'] = args.batch_size

    device = get_device(cfg, args.device)
    data = build_data(cfg)

    ckpt_path = resolve_ckpt_path(cfg, args.ckpt)
    model = load_model(cfg, ckpt_path, device)

    batch_size = cfg['data'].get('batch_size', 128)
    class_names = getattr(data, 'class_names', None) or \
                  [str(i) for i in range(cfg['model'].get('params', {}).get('num_classes', 10))]

    model_name = cfg['model']['name']

    # ---- 单图预测模式 ----
    if args.predict_single is not None:
        out_root = Path(cfg.get('output', {}).get('root', '../result/week07'))
        chart_dir = out_root / cfg.get('output', {}).get('chart_dir', 'charts')
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = chart_dir / f"predict_{model_name}_idx{args.predict_single}_{ts}.jpg"
        predict_single(model, data, args.predict_single, device, save_path)
        return

    # ---- 全量评估模式 ----
    test_loader = data.get_test_loader()
    avg_loss, accuracy, y_true, y_pred, indices = evaluate(model, test_loader, device)

    print("\n" + "=" * 50)
    print(f"模型: {model_name}")
    print(f"权重: {ckpt_path}")
    print(f"测试集: {cfg['data'].get('name', 'unknown')} ({len(test_loader.dataset)} 样本)")
    print("-" * 50)
    print(f"Test Loss:     {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print("=" * 50)

    out_root = Path(cfg.get('output', {}).get('root', '../result/week07'))
    chart_dir = out_root / cfg.get('output', {}).get('chart_dir', 'charts')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    test_cfg = cfg.get('test', {})

    # 混淆矩阵
    if test_cfg.get('plot_confusion_matrix', True):
        cm_path = chart_dir / f"confusion_matrix_{model_name}_{ts}.jpg"
        plot_confusion_matrix_top_n(y_true, y_pred, class_names, cm_path, top_n=20)

    # ---- 类别级准确率（CSV 数据文件 + 条形图）----
    if test_cfg.get('per_class_csv', True) or test_cfg.get('per_class_chart', True):
        log_dir = out_root / cfg.get('output', {}).get('log_dir', 'logs')
        per_csv = log_dir / f"per_class_metrics_{model_name}_{ts}.csv" \
            if test_cfg.get('per_class_csv', True) else None
        per_fig = chart_dir / f"per_class_accuracy_{model_name}_{ts}.jpg" \
            if test_cfg.get('per_class_chart', True) else None
        export_per_class_metrics(y_true, y_pred, class_names,
                                 csv_path=per_csv, fig_path=per_fig)

    # ---- 错误样例展示（图片 + 真实结果 + 预测结果）----
    num_errors = args.num_errors if args.num_errors is not None \
        else test_cfg.get('error_samples_num', 16)
    if test_cfg.get('plot_error_samples', True) and num_errors > 0:
        error_indices = [i for i in indices if y_true[i] != y_pred[i]]
        print(f"\n错误样本总数: {len(error_indices)}，展示前 {min(num_errors, len(error_indices))} 张")
        err_path = chart_dir / f"error_samples_{model_name}_{ts}.jpg"
        plot_prediction_grid(
            data.test_dataset, error_indices[:num_errors],
            y_true, y_pred, class_names, err_path,
            data_cfg=cfg['data'],
            title=f"Error Samples (true vs pred) | {model_name}",
            only_errors=True)

    # ---- 预测展示（前 N 个样本，绿=正确 红=错误）----
    num_preds = args.num_preds if args.num_preds is not None \
        else test_cfg.get('prediction_samples_num', 16)
    if test_cfg.get('plot_prediction_samples', True) and num_preds > 0:
        show_indices = indices[:num_preds]
        pred_path = chart_dir / f"prediction_samples_{model_name}_{ts}.jpg"
        plot_prediction_grid(
            data.test_dataset, show_indices,
            y_true, y_pred, class_names, pred_path,
            data_cfg=cfg['data'],
            title=f"Prediction Samples | {model_name} | acc={accuracy:.3f}")


if __name__ == '__main__':
    main()
