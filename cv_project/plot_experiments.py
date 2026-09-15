"""
plot_experiments.py - 第三步：只读数据文件绘图（不训练、不重新推理）

数据来源（全部是第二步 experiments.py 的产物）：
  跨组对比图  : 读全局总表 results_root/summary.csv，按 run_group 筛选、按参数排序
  逐轮曲线    : 读各 run 目录的 epochs.csv
  类别级细节  : 读各 run 目录的 per_class_metrics.csv

输出到 results_root/_plots/：
  overview_all_runs.jpg            全部 run 的 test_acc 排名（一张总览）
  bar_<group>_acc.jpg / _loss.jpg  组内各参数的最终指标柱状对比
  curves_<group>.jpg               组内各 run 的逐 epoch loss/acc 曲线叠加
  perclass_<group>.jpg             组内最差 15 类的 recall 对比（文件存在时）

用法:
  python plot_experiments.py
  python plot_experiments.py --only-groups lr,pretrained
"""
import os
import csv
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 字符串型实验变量的排序口径（按实验语义而非字母序）
RANK_MAPS = {
    'pretrained': {'False': 0, 'false': 0, 'True': 1, 'true': 1},
    'augmentation': {'none': 0, 'light': 1, 'standard': 2},
    'finetune_mode': {'feature_extract': 0, 'full': 1},
}


# ============================================================
# 读取（只读）
# ============================================================
def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_summary(summary_path):
    """读全局总表，返回成功 run 的行列表（保持文件中的顺序）。"""
    if not summary_path.exists():
        raise FileNotFoundError(f"全局总表不存在: {summary_path}，请先运行 experiments.py")
    with open(summary_path, 'r', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    ok = [r for r in rows if r.get('status') == 'success']
    failed = [r for r in rows if r.get('status') != 'success']
    if failed:
        print(f"[!] 总表中有 {len(failed)} 个失败 run，绘图时已跳过")
    return ok


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_epochs_csv(path):
    """读 run 目录的 epochs.csv，返回列名字典。"""
    out = {}
    with open(path, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                out.setdefault(k, []).append(to_float(v))
    return out


def read_per_class_csv(path):
    """读 per_class_metrics.csv，返回 {class_index: {'name':..,'recall':..}}。"""
    stats = {}
    with open(path, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            recall = to_float(row.get('recall'))
            stats[int(row['class_index'])] = {
                'name': row['class_name'],
                'recall': recall,
            }
    return stats


def resolve_run_dir(results_root, rel_dir):
    """把总表中的 run_dir 解析为实际数据所在目录。

    复用产生的软引用目录里没有 epochs.csv/predictions.csv，只有 reuse_from.yaml
    指针；检测到指针时跳转到源 run 目录。--copy-reuse 复制模式无需跳转。
    """
    p = results_root / rel_dir
    ref = p / 'reuse_from.yaml'
    if ref.exists():
        d = load_yaml(ref)
        return results_root / d['source_run_dir']
    return p


# ============================================================
# 排序：按组内变化的参数排序
# ============================================================
def varying_column(rows):
    """找出组内实际发生变化的参数列（lr / pretrained / augmentation / finetune_mode）。"""
    candidates = ['lr', 'pretrained', 'augmentation', 'finetune_mode']
    for col in candidates:
        vals = {r.get(col) for r in rows}
        if len(vals) > 1:
            return col
    return None


def group_sort_key(row, sort_col):
    """组内排序：数值型参数按数值；枚举型按 RANK_MAPS；其余按 tag。"""
    if sort_col is None:
        return row['tag']
    v = row.get(sort_col, '')
    if sort_col == 'lr':
        return to_float(v) if to_float(v) is not None else 0.0
    return RANK_MAPS.get(sort_col, {}).get(v, 99)


def sort_group_rows(rows):
    sort_col = varying_column(rows)
    return sorted(rows, key=lambda r: group_sort_key(r, sort_col)), sort_col


# ============================================================
# 图 1：全部 run 总览（读全局总表）
# ============================================================
def plot_overview(rows, out_path):
    rows = sorted(rows, key=lambda r: to_float(r['eval_test_acc']) or -1)
    labels = [f"{r['run_group']}/{r['tag']}" for r in rows]
    accs = [to_float(r['eval_test_acc']) for r in rows]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(rows) + 1)))
    colors = plt.cm.viridis([i / max(len(accs) - 1, 1) for i in range(len(accs))])
    ax.barh(range(len(rows)), accs, color=colors)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Test Accuracy (best-model eval)')
    ax.set_title(f'All Runs Overview ({len(rows)} runs)')
    for i, v in enumerate(accs):
        ax.text(v + 0.003, i, f'{v:.3f}', va='center', fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] {out_path.name}")


# ============================================================
# 图 2：组内柱状对比（读全局总表，按变化参数排序）
# ============================================================
def plot_group_bars(rows, group, out_dir):
    rows_sorted, sort_col = sort_group_rows(rows)
    x = [r['tag'] for r in rows_sorted]
    accs = [to_float(r['eval_test_acc']) for r in rows_sorted]
    losses = [to_float(r['eval_test_loss']) for r in rows_sorted]

    for metric, vals, fname, title in [
        ('test_acc', accs, f'bar_{group}_acc.jpg',
         f'{group}: test accuracy by {sort_col or "tag"}'),
        ('test_loss', losses, f'bar_{group}_loss.jpg',
         f'{group}: test loss by {sort_col or "tag"}'),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(x, vals, color='#4C72B0')
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_ylim(0, 1.0) if metric == 'test_acc' else None
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[OK] {fname}")


# ============================================================
# 图 3：组内逐 epoch 曲线叠加（读各 run 的 epochs.csv）
# ============================================================
def plot_group_curves(rows, group, results_root, out_dir):
    curves = []
    for r in rows:
        data_dir = resolve_run_dir(results_root, r['run_dir'])
        p = data_dir / 'epochs.csv'
        if p.exists():
            curves.append((r['tag'], read_epochs_csv(p)))
    if not curves:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for tag, h in curves:
        ep = h['epoch']
        axes[0].plot(ep, h['test_loss'], '--', marker='o', ms=3, label=tag)
        axes[1].plot(ep, h['test_acc'], '--', marker='o', ms=3, label=tag)
    axes[0].set_xlabel('epoch'); axes[0].set_ylabel('test loss')
    axes[0].set_title(f'{group}: test loss / epoch')
    axes[1].set_xlabel('epoch'); axes[1].set_ylabel('test acc')
    axes[1].set_title(f'{group}: test acc / epoch')
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = out_dir / f'curves_{group}.jpg'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] {out.name}")


# ============================================================
# 图 4：组内类别级 recall 对比（读各 run 的 per_class_metrics.csv）
# ============================================================
def plot_group_per_class(rows, group, results_root, out_dir, top_k=15):
    per_run = []
    for r in rows:
        data_dir = resolve_run_dir(results_root, r['run_dir'])
        p = data_dir / 'per_class_metrics.csv'
        if p.exists():
            per_run.append((r['tag'], read_per_class_csv(p)))
    if len(per_run) < 2:
        return

    # 以第一个 run 的 recall 找最差的 top_k 类
    first_tag, first_stats = per_run[0]
    valid = [(idx, s['recall']) for idx, s in first_stats.items()
             if s['recall'] is not None]
    valid.sort(key=lambda kv: kv[1])
    worst = [idx for idx, _ in valid[:top_k]]
    if not worst:
        return

    names = [first_stats[idx]['name'] for idx in worst]
    import numpy as np
    x = np.arange(len(worst))
    width = 0.8 / len(per_run)

    fig, ax = plt.subplots(figsize=(max(10, len(worst) * 0.7), 5))
    for i, (tag, stats) in enumerate(per_run):
        vals = [(stats.get(idx, {}).get('recall') or 0.0) for idx in worst]
        ax.bar(x + (i - (len(per_run) - 1) / 2) * width, vals,
               width, label=tag)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha='right', fontsize=7)
    ax.set_ylabel('recall')
    ax.set_ylim(0, 1)
    ax.set_title(f'{group}: per-class recall on {top_k} worst classes '
                 f'(ranked by {first_tag})')
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = out_dir / f'perclass_{group}.jpg'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] {out.name}")


# ============================================================
# 主入口
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description='实验结果绘图（第三步，只读文件）')
    p.add_argument('--config', default='experiment_config.yaml')
    p.add_argument('--only-groups', help='逗号分隔，只画指定组，如 lr,pretrained')
    return p.parse_args()


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    args = parse_args()

    exp_cfg = load_yaml(args.config)['experiment']
    results_root = Path(exp_cfg['results_root'])
    summary_path = results_root / exp_cfg.get('global_summary', 'summary.csv')
    out_dir = results_root / '_plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_summary(summary_path)
    only = set(g.strip() for g in args.only_groups.split(',')) if args.only_groups else None

    # 按组分桶（保持总表顺序）
    groups = {}
    for r in rows:
        if only and r['run_group'] not in only:
            continue
        groups.setdefault(r['run_group'], []).append(r)

    print(f"读取总表: {summary_path}（{len(rows)} 个成功 run，"
          f"{len(groups)} 个组）\n绘图输出: {out_dir}\n")

    # 总览
    plot_overview(rows, out_dir / 'overview_all_runs.jpg')

    # 各组细节
    for group, grows in groups.items():
        if len(grows) >= 2:
            plot_group_bars(grows, group, out_dir)
            plot_group_curves(grows, group, results_root, out_dir)
            plot_group_per_class(grows, group, results_root, out_dir)
        else:
            print(f"-- 组 {group} 仅 {len(grows)} 个成功 run，跳过组内对比图")

    print("\n完成。所有图均由 summary.csv / epochs.csv / per_class_metrics.csv 生成，"
          "未执行任何训练或推理。")


if __name__ == '__main__':
    main()
