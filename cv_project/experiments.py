"""
experiments.py - 第二步：按实验总配置逐组执行训练与评估

三阶段中的「执行阶段」：
  - 读 experiment_config.yaml（计划）+ config.yaml（基线完整配置）
  - 每个 run 展开为一份独立完整配置（基线 + 点分覆盖），分配独立输出目录
  - 训练过程只写数据文件，不画图；训练后加载 best_model.pth 做评估
  - 每个 run 目录写：config_used.yaml / epochs.csv / summary.json /
                    predictions.csv / per_class_metrics.csv / checkpoints/
  - 每完成一个 run，向全局总表 summary.csv 追加一行（失败也记录）

结果复用（相同训练配置不重复训练）:
  - 每个 run 先算「训练配置指纹」（只含影响训练结果的字段并补齐默认值，
    因此显式 aug=none 与缺省、显式 pretrained=true 与基线都判为相同）
  - 若已有同指纹的成功 run（如 lr_0.001 ≡ baseline），默认只写一个
    reuse_from.yaml 指针直接引用，不再训练；指标照常进入全局总表
  - --copy-reuse 改为复制全套产物（best_model.pth 同盘优先硬链接，省磁盘）
  - --no-reuse 强制重新训练

用法:
  python experiments.py --list                      # 只展开计划，不训练
  python experiments.py                             # 执行全部组（自动复用同配置）
  python experiments.py --only-group lr             # 只跑某组
  python experiments.py --only-group lr --only-tag lr_0.0001
  python experiments.py --epochs 1 --device cpu     # 冒烟：强制 1 epoch
  python experiments.py --overwrite                 # 覆盖已有运行目录
  python experiments.py --no-reuse                  # 禁止复用，全部重训
  python experiments.py --copy-reuse                # 命中复用时复制产物而非软引用
"""
import os
import csv
import json
import copy
import time
import shutil
import hashlib
import argparse
import platform
from pathlib import Path
from datetime import datetime

import torch
import yaml

import train as train_mod
from dataset import build_data
from model import build_model
from train import Trainer, set_seed
# test.py 的评估与类别统计函数直接复用（不重写推理逻辑）
from test import evaluate, export_per_class_metrics

# 实验执行阶段只写数据、不画图：关闭 Trainer 的动画画板
# （HAS_IPYTHON 为 True 时 ProgressBoard 每个 batch 都会创建 figure 对象）
train_mod.HAS_IPYTHON = False


# ============================================================
# 配置加载
# ============================================================
def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description='实验批量执行（第二步）')
    p.add_argument('--config', default='experiment_config.yaml')
    p.add_argument('--list', action='store_true',
                   help='只展开并打印实验计划，不训练')
    p.add_argument('--only-group', help='只执行指定组，如 lr / pretrained / augmentation')
    p.add_argument('--only-tag', help='只执行指定 tag（需配合 --only-group 或全局唯一）')
    p.add_argument('--overwrite', action='store_true',
                   help='run 目录已存在时覆盖；默认追加时间戳后缀保证唯一定位')
    p.add_argument('--no-reuse', action='store_true',
                   help='禁止结果复用，相同训练配置也强制重新训练')
    p.add_argument('--copy-reuse', action='store_true',
                   help='命中复用时复制源 run 全套产物（.pth 优先硬链接）；'
                        '默认只写 reuse_from.yaml 软引用')
    return p.parse_args()


# ============================================================
# 计划展开（纯配置操作，不涉及任何训练）
# ============================================================
def apply_dotted_override(cfg, dotted_key, value):
    """把 'train.lr' 这类点分路径写入 cfg。"""
    keys = dotted_key.split('.')
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def expand_plan(exp_cfg, base_cfg):
    """把 experiment_config 的 groups 展开成 run 列表。

    每个 run: {group, tag, description, overrides, cfg}
    cfg 是基线的深拷贝 + 覆盖项，组间互不影响。
    """
    runs = []
    for grp in exp_cfg['experiment']['groups']:
        group_name = grp['group']
        desc = grp.get('description', '')
        for run in grp.get('runs', []):
            tag = run['tag']
            overrides = run.get('overrides', {}) or {}

            cfg = copy.deepcopy(base_cfg)
            for dotted_key, value in overrides.items():
                apply_dotted_override(cfg, dotted_key, value)

            runs.append({
                'group': group_name,
                'tag': tag,
                'description': desc,
                'overrides': overrides,
                'cfg': cfg,
            })
    return runs


def print_plan(runs, results_root):
    print(f"实验根目录: {results_root}")
    print(f"共 {len(runs)} 个 run:\n")
    cur_group = None
    for r in runs:
        if r['group'] != cur_group:
            cur_group = r['group']
            print(f"[{cur_group}] {r['description']}")
        ov = r['overrides']
        ov_str = ', '.join(f'{k}={v}' for k, v in ov.items()) if ov else '(基线，无覆盖)'
        print(f"    {r['tag']:<20} {ov_str}")
    print()


# ============================================================
# 全局总表
# ============================================================
SUMMARY_COLUMNS = [
    'run_group', 'tag', 'run_dir', 'status', 'finished_at',
    'fingerprint', 'reused_from',
    'model', 'dataset', 'optimizer', 'lr', 'weight_decay',
    'batch_size', 'max_epoch', 'epochs_run',
    'pretrained', 'finetune_mode',
    'augmentation', 'seed',
    'best_epoch', 'eval_test_loss', 'eval_test_acc',
    'macro_recall', 'weighted_recall',
    'history_min_test_loss', 'history_max_test_acc',
    'total_time_s', 'error',
]


def append_summary_row(summary_path, row):
    """向全局总表追加一行；表不存在则先写表头。崩溃后重跑也不丢历史。"""
    is_new = not summary_path.exists()
    with open(summary_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, '') for k in SUMMARY_COLUMNS})


# ============================================================
# 训练配置指纹 + 结果复用（相同配置不重复训练）
# ============================================================
def _norm_list(v):
    return list(v) if isinstance(v, (list, tuple)) else v


def training_fingerprint(cfg):
    """提取「影响训练结果」的配置，补齐默认值后哈希成指纹。

    归一化补默认值是关键：基线缺省 data.augmentation（实际 none）与显式
    覆盖 data.augmentation=none 必须得到同一指纹。
    排除项：device/num_workers/pin_memory/output/logging——不改变训练结果。
    """
    data_cfg = cfg.get('data', {})
    model_cfg = cfg.get('model', {})
    train_cfg = cfg.get('train', {})
    key = {
        'seed': cfg.get('seed', 42),
        'deterministic': cfg.get('deterministic', True),
        'data': {
            'name': data_cfg.get('name'),
            'root': str(data_cfg.get('root', './data')),
            'resize': list(data_cfg.get('resize', [224, 224])),
            'rgb_expand': data_cfg.get('rgb_expand', False),
            'mean': _norm_list(data_cfg.get('mean', [0.5])),
            'std': _norm_list(data_cfg.get('std', [0.5])),
            'augmentation': data_cfg.get('augmentation', 'none'),
            'batch_size': data_cfg.get('batch_size', 128),
            'download': data_cfg.get('download', True),
            'train_dir': data_cfg.get('train_dir'),
            'val_dir': data_cfg.get('val_dir'),
        },
        'model': {
            'name': model_cfg.get('name'),
            'params': model_cfg.get('params', {}),
            'pretrained_path': model_cfg.get('pretrained_path'),
            # 微调模式影响可训练参数范围，必须参与指纹
            'finetune_mode': model_cfg.get('finetune_mode', 'full'),
        },
        'train': {
            'max_epoch': train_cfg.get('max_epoch', 10),
            'lr': train_cfg.get('lr', 0.001),
            'optimizer': train_cfg.get('optimizer', 'sgd'),
            'momentum': train_cfg.get('momentum', 0.9),
            'weight_decay': train_cfg.get('weight_decay', 0.0),
            'early_stopping': train_cfg.get('early_stopping', {}),
        },
    }
    raw = json.dumps(key, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]


def find_reusable_run(results_root, fingerprint):
    """扫描所有 run 目录的 summary.json，返回同指纹的最新成功源目录。

    引用目录（status=reused）不写入指纹，不会形成引用链。
    返回 Path（源 run 目录）或 None。
    """
    candidates = []
    for sj in results_root.glob('*/*/summary.json'):
        try:
            with open(sj, 'r', encoding='utf-8') as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if d.get('status') == 'success' and d.get('fingerprint') == fingerprint:
            candidates.append((sj.stat().st_mtime, sj.parent))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1] if candidates else None


def _link_or_copy(src, dst):
    """同盘优先硬链接（省磁盘），失败（跨卷/权限）退回复制。"""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_run_artifacts(src_dir, dst_dir):
    """复制源 run 的全部产物到目标目录；.pth 优先硬链接。"""
    for item in src_dir.iterdir():
        if item.is_dir():
            for f in item.rglob('*'):
                if f.is_file():
                    rel = f.relative_to(src_dir)
                    target = dst_dir / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if f.suffix == '.pth':
                        _link_or_copy(f, target)
                    else:
                        shutil.copy2(f, target)
        elif item.name != 'reuse_from.yaml':
            shutil.copy2(item, dst_dir / item.name)


# ============================================================
# run 目录内文件写入
# ============================================================
def write_config_snapshot(run_dir, cfg):
    with open(run_dir / 'config_used.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def write_epochs_csv(run_dir, history):
    path = run_dir / 'epochs.csv'
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'train_loss', 'train_acc',
                    'test_loss', 'test_acc', 'lr', 'epoch_time'])
        for i in range(len(history['train_loss'])):
            w.writerow([i + 1,
                        history['train_loss'][i], history['train_acc'][i],
                        history['test_loss'][i], history['test_acc'][i],
                        history['lr'][i], history['epoch_time'][i]])


def write_predictions_csv(run_dir, indices, y_true, y_pred, class_names):
    path = run_dir / 'predictions.csv'
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['index', 'true_id', 'pred_id', 'correct',
                    'true_name', 'pred_name'])
        for idx, t, p in zip(indices, y_true, y_pred):
            w.writerow([idx, t, p, int(t == p),
                        class_names[t], class_names[p]])


def write_summary_json(run_dir, payload):
    with open(run_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def macro_weighted_recall(y_true, y_pred, num_classes):
    """macro / weighted recall（类别级统计汇总）。"""
    try:
        from sklearn.metrics import precision_recall_fscore_support
    except ImportError:
        return None, None
    labels = list(range(num_classes))
    _, macro_r, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0)
    _, weighted_r, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    return float(macro_r), float(weighted_r)


# ============================================================
# 单个 run 的执行
# ============================================================
def resolve_run_dir(results_root, group, tag, overwrite):
    """分配独立运行目录；重名时默认加时间戳后缀，保证唯一定位。"""
    run_dir = results_root / group / tag
    if run_dir.exists() and not overwrite:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = results_root / group / f"{tag}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def reuse_run(run, run_dir, src_dir, results_root, fingerprint, copy_mode):
    """命中相同训练配置：不训练，直接引用/复制源 run 产物，并向总表追加一行。"""
    group, tag = run['group'], run['tag']
    cfg = run['cfg']
    src_rel = str(src_dir.relative_to(results_root)).replace('\\', '/')
    finished_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 本组实际使用配置（输出路径指向本 run 目录），供复现
    cfg = copy.deepcopy(cfg)
    cfg['output']['root'] = str(run_dir)
    write_config_snapshot(run_dir, cfg)

    with open(src_dir / 'summary.json', 'r', encoding='utf-8') as f:
        src_payload = json.load(f)

    if copy_mode:
        copy_run_artifacts(src_dir, run_dir)
        print(f"[REUSE-COPY] [{group}/{tag}] 复制自 {src_rel}")
    else:
        # 软引用：只写指针，绘图脚本据此跳转到源目录读 epochs/predictions 等
        with open(run_dir / 'reuse_from.yaml', 'w', encoding='utf-8') as f:
            yaml.safe_dump({
                'status': 'reused',
                'source_group': src_payload.get('group'),
                'source_tag': src_payload.get('tag'),
                'source_run_dir': src_rel,
                'fingerprint': fingerprint,
                'reused_at': finished_at,
            }, f, allow_unicode=True, sort_keys=False)
        print(f"[REUSE] [{group}/{tag}] 命中相同训练配置，直接引用 {src_rel}（未训练）")

    # 本目录 summary.json 标记为 reused（无 fingerprint，不会被当作复用源）
    write_summary_json(run_dir, {
        'group': group, 'tag': tag, 'status': 'reused',
        'reuse_from': src_rel,
        'fingerprint': fingerprint,
        'finished_at': finished_at,
    })

    p = src_payload.get('params', {})
    be = src_payload.get('best_model_eval', {})
    h = src_payload.get('history', {})
    row = {
        'run_group': group, 'tag': tag,
        'run_dir': str(run_dir.relative_to(results_root)).replace('\\', '/'),
        'status': 'success',          # 对绘图/统计而言与成功 run 等价
        'finished_at': finished_at,
        'fingerprint': fingerprint,
        'reused_from': src_rel,
        'model': p.get('model'), 'dataset': p.get('dataset'),
        'optimizer': p.get('optimizer'), 'lr': p.get('lr'),
        'weight_decay': p.get('weight_decay'),
        'batch_size': p.get('batch_size'), 'max_epoch': p.get('max_epoch'),
        'pretrained': p.get('pretrained'),
        'finetune_mode': p.get('finetune_mode'),
        'augmentation': p.get('augmentation'), 'seed': p.get('seed'),
        'epochs_run': src_payload.get('epochs_run'),
        'best_epoch': src_payload.get('best_epoch'),
        'eval_test_loss': be.get('test_loss'),
        'eval_test_acc': be.get('test_acc'),
        'macro_recall': src_payload.get('macro_recall'),
        'weighted_recall': src_payload.get('weighted_recall'),
        'history_min_test_loss': h.get('min_test_loss'),
        'history_max_test_acc': h.get('max_test_acc'),
        'total_time_s': 0.0,
        'error': '',
    }
    # 数值统一成 6 位（与训练路径的总表口径一致）
    for k in ('eval_test_loss', 'eval_test_acc', 'macro_recall',
              'weighted_recall', 'history_min_test_loss', 'history_max_test_acc'):
        if isinstance(row[k], float):
            row[k] = round(row[k], 6)
    append_summary_row(results_root / SUMMARY_FILENAME, row)
    return True


def run_one(run, results_root, overwrite, verbose=True,
            allow_reuse=True, copy_reuse=False):
    """执行单个 run：命中同指纹已完成 run 则引用；否则
    训练 → 加载最佳权重评估 → 写全套数据文件 → 追加全局总表。"""
    group, tag = run['group'], run['tag']
    cfg = run['cfg']

    run_dir = resolve_run_dir(results_root, group, tag, overwrite)

    # ---- 计算训练指纹（归一化后），查找可复用的成功 run ----
    # 注意：不带 --overwrite 重跑时本 run 会分到时间戳后缀目录，与源目录
    # 路径不同，默认复用上次结果（这正是本特性的目的）；--overwrite 时目标
    # 目录即源目录，路径相同，跳过复用执行重训。强制重训用 --no-reuse。
    fingerprint = training_fingerprint(cfg)
    src_dir = find_reusable_run(results_root, fingerprint) if allow_reuse else None
    if src_dir is not None and src_dir.resolve() != run_dir.resolve():
        return reuse_run(run, run_dir, src_dir, results_root,
                         fingerprint, copy_mode=copy_reuse)

    print("\n" + "#" * 70)
    print(f"# RUN [{group}/{tag}]  ->  {run_dir}")
    print("#" * 70)

    finished_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # run_dir 存相对 results_root 的路径（group/tag[+时间戳]），绘图脚本按约定拼接
    rel_run_dir = str(run_dir.relative_to(results_root)).replace('\\', '/')
    base_row = {
        'run_group': group, 'tag': tag,
        'run_dir': rel_run_dir,
        'finished_at': finished_at,
        'fingerprint': fingerprint,
        'reused_from': '',
        'model': cfg['model'].get('name'),
        'dataset': cfg['data'].get('name'),
        'optimizer': cfg['train'].get('optimizer', 'sgd'),
        'lr': cfg['train'].get('lr'),
        'weight_decay': cfg['train'].get('weight_decay', 0.0),
        'batch_size': cfg['data'].get('batch_size'),
        'max_epoch': cfg['train'].get('max_epoch'),
        'pretrained': cfg['model'].get('params', {}).get('pretrained'),
        'finetune_mode': cfg['model'].get('finetune_mode', 'full'),
        'augmentation': cfg['data'].get('augmentation', 'none'),
        'seed': cfg.get('seed', 42),
    }

    total_start = time.time()
    try:
        # ---- 复现种子 ----
        set_seed(cfg.get('seed', 42),
                 cfg.get('deterministic', True),
                 cfg.get('benchmark', False))

        # ---- 输出重定向到本 run 目录；关闭 Trainer 自带日志/画图 ----
        cfg['output']['root'] = str(run_dir)
        cfg.setdefault('logging', {})
        cfg['logging'].update(csv_log=False, run_meta=False, plot_curves=False)

        # ---- 0. 配置副本（复现依据）----
        write_config_snapshot(run_dir, cfg)

        # ---- 1. 数据 / 模型 / 训练（只写 checkpoint + 内存历史，不画图）----
        data = build_data(cfg)
        model = build_model(cfg)
        trainer = Trainer(model, data, cfg, model_name=tag, verbose=verbose)
        history = trainer.train(name=tag)

        # ---- 2. epochs.csv（每个 epoch 的指标）----
        write_epochs_csv(run_dir, history)

        # ---- 3. 加载最佳权重再评估（保证预测来自 best_model，而非最后一轮）----
        ckpt_path = run_dir / 'checkpoints' / cfg['output'].get(
            'best_model_name', 'best_model.pth')
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=trainer.device,
                               weights_only=True)
            trainer.model.load_state_dict(state)

        avg_loss, acc, y_true, y_pred, indices = evaluate(
            trainer.model, trainer.test_loader, trainer.device)
        class_names = data.class_names

        # ---- 4. predictions.csv（逐样本预测结果）----
        write_predictions_csv(run_dir, indices, y_true, y_pred, class_names)

        # ---- 5. per_class_metrics.csv（类别级统计；fig_path=None 不画图）----
        export_per_class_metrics(
            y_true, y_pred, class_names,
            csv_path=run_dir / 'per_class_metrics.csv',
            fig_path=None)

        # ---- 6. 汇总指标 ----
        macro_r, weighted_r = macro_weighted_recall(
            y_true, y_pred, len(class_names))
        total_time = time.time() - total_start
        summary_payload = {
            'group': group, 'tag': tag, 'run_dir': str(run_dir),
            'finished_at': finished_at,
            'fingerprint': fingerprint,
            'overrides': run['overrides'],
            'params': {k: base_row[k] for k in
                       ('model', 'dataset', 'optimizer', 'lr', 'weight_decay',
                        'batch_size', 'max_epoch', 'pretrained',
                        'finetune_mode', 'augmentation', 'seed')},
            'epochs_run': len(history['train_loss']),
            'best_epoch': trainer.best_epoch,
            'best_model_eval': {'test_loss': avg_loss, 'test_acc': acc},
            'macro_recall': macro_r,
            'weighted_recall': weighted_r,
            'history': {
                'min_test_loss': min(history['test_loss']),
                'max_test_acc': max(history['test_acc']),
                'final_test_loss': history['test_loss'][-1],
                'final_test_acc': history['test_acc'][-1],
            },
            'torch_version': torch.__version__,
            'python_version': platform.python_version(),
            'total_time_s': round(total_time, 2),
            'status': 'success',
        }
        write_summary_json(run_dir, summary_payload)

        # ---- 7. 追加全局总表 ----
        row = dict(base_row)
        row.update({
            'status': 'success',
            'epochs_run': len(history['train_loss']),
            'best_epoch': trainer.best_epoch,
            'eval_test_loss': round(avg_loss, 6),
            'eval_test_acc': round(acc, 6),
            'macro_recall': '' if macro_r is None else round(macro_r, 6),
            'weighted_recall': '' if weighted_r is None else round(weighted_r, 6),
            'history_min_test_loss': round(min(history['test_loss']), 6),
            'history_max_test_acc': round(max(history['test_acc']), 6),
            'total_time_s': round(total_time, 2),
            'error': '',
        })
        append_summary_row(results_root / SUMMARY_FILENAME, row)

        print(f"[OK] [{group}/{tag}] eval_loss={avg_loss:.4f} "
              f"eval_acc={acc:.4f} macro_recall={macro_r}")

        # 释放显存，避免组间累积
        del trainer, model, data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True

    except Exception as e:
        import traceback
        traceback.print_exc()
        total_time = time.time() - total_start
        row = dict(base_row)
        row.update({
            'status': 'failed',
            'total_time_s': round(total_time, 2),
            'error': f'{type(e).__name__}: {e}',
        })
        append_summary_row(results_root / SUMMARY_FILENAME, row)
        write_summary_json(run_dir, {
            'group': group, 'tag': tag, 'status': 'failed',
            'finished_at': finished_at,
            'error': row['error'],
            'total_time_s': round(total_time, 2),
        })
        print(f"[FAIL] [{group}/{tag}] {row['error']}")
        return False


SUMMARY_FILENAME = 'summary.csv'


# ============================================================
# 主入口
# ============================================================
def main():
    # 切换到脚本所在目录，保证相对路径一致
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    args = parse_args()
    exp_cfg = load_yaml(args.config)
    base_cfg = load_yaml(exp_cfg['experiment']['base_config'])

    results_root = Path(exp_cfg['experiment']['results_root'])
    results_root.mkdir(parents=True, exist_ok=True)

    runs = expand_plan(base_cfg=base_cfg, exp_cfg=exp_cfg)

    # 过滤
    if args.only_group:
        runs = [r for r in runs if r['group'] == args.only_group]
    if args.only_tag:
        runs = [r for r in runs if r['tag'] == args.only_tag]

    # --list：只打印计划，不训练
    if args.list:
        print_plan(runs, results_root)
        return

    global SUMMARY_FILENAME
    SUMMARY_FILENAME = exp_cfg['experiment'].get('global_summary', 'summary.csv')

    print_plan(runs, results_root)
    if not args.no_reuse:
        print("结果复用已开启：相同训练配置的 run 将直接引用已有结果（--no-reuse 关闭）\n")
    trained, reused, fail = 0, 0, 0
    for i, run in enumerate(runs, 1):
        print(f"\n>>> ({i}/{len(runs)}) 开始执行: {run.get('tag', run)}")
        try:
            success = run_one(run, results_root, overwrite=args.overwrite,
                              allow_reuse=not args.no_reuse,
                              copy_reuse=args.copy_reuse)
        except Exception as e:
            print(f"[!] 执行失败: {e}")
            success = False

        if not success:
            fail += 1
            continue
        row = _read_last_summary_row(results_root / SUMMARY_FILENAME)
        if row and row.get('reused_from'):
            reused += 1
        else:
            trained += 1

    print("\n" + "=" * 70)
    print(f"全部结束：实际训练 {trained}，复用 {reused}，失败 {fail}")
    print(f"全局总表: {results_root / SUMMARY_FILENAME}")
    print("绘图请运行: python plot_experiments.py")
    print("=" * 70)


def _read_last_summary_row(summary_path):
    """读总表最后一行（用于判断刚完成的 run 是训练还是复用）。"""
    try:
        with open(summary_path, 'r', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        return rows[-1] if rows else None
    except OSError:
        return None


if __name__ == '__main__':
    main()
