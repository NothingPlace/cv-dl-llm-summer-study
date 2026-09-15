"""
models.py - 统一的模型管理模块

包含：
1. 模型注册表机制
2. 模型注册装饰器
3. 模型构建工厂函数
4. 权重初始化工具
5. 预训练权重加载工具
6. 多个模型定义（AlexNet、ResNet、MLP 等）
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import importlib


# ============================================================================
# 第一部分：注册表与工厂函数
# ============================================================================

MODEL_REGISTRY: Dict[str, Any] = {}

def register_model(name: str):
    """
    模型注册装饰器
    
    用法:
        @register_model("alexnet")
        class AlexNet(nn.Module):
            ...
    """
    def decorator(cls):
        if not issubclass(cls, nn.Module):
            raise TypeError(f"{name} 不是 nn.Module 子类")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def build_model(cfg: dict) -> nn.Module:
    """
    根据配置构建模型
    
    配置格式:
        model:
          name: "alexnet"           # 模型名称（必须在注册表中）
          params:                   # 模型参数（可选）
            num_classes: 10
          pretrained_path: null     # 预训练权重路径（可选）
    
    Args:
        cfg: 配置字典，必须包含 model.name
        
    Returns:
        nn.Module: 模型实例
    """
    model_cfg = cfg.get("model", {})
    if not model_cfg:
        raise ValueError("配置中缺少 'model' 字段")
    
    model_name = model_cfg.get("name")
    if not model_name:
        raise ValueError("配置中缺少 'model.name' 字段")
    
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"未知模型: '{model_name}'，可用模型: {list(MODEL_REGISTRY.keys())}"
        )
    
    # 构建模型（值为 None 的参数过滤掉：架构切换时用 null 取消
    # 该模型不支持的参数，如 AlexNet 不需要 pretrained）
    model_params = {k: v for k, v in model_cfg.get("params", {}).items()
                    if v is not None}
    model = MODEL_REGISTRY[model_name](**model_params)

    # 加载预训练权重（可选）
    pretrained_path = model_cfg.get("pretrained_path")
    if pretrained_path:
        load_pretrained(model, pretrained_path)

    # ---- 微调模式：feature_extract / full（默认 full 保持基线语义）----
    ft_mode = model_cfg.get("finetune_mode", "full")
    if ft_mode != "full":
        ft_stats = apply_finetune_mode(model, ft_mode)
        print(f"[build_model] finetune_mode={ft_mode}"
              f": 可训练 {ft_stats['trainable_params']:,}"
              f" / {ft_stats['trainable_params'] + ft_stats['frozen_params']:,}"
              f" ({ft_stats['trainable_ratio']*100:.1f}%)")

    return model

def list_available_models() -> list:
    """列出所有已注册的模型名称"""
    return list(MODEL_REGISTRY.keys())


# ============================================================================
# 第二部分：权重初始化与预训练加载工具
# ============================================================================

# Lazy 层（LazyConv2d/LazyLinear）在首次 forward 前没有 shape，
# weight 是 UninitializedParameter，此时不能做原地初始化；
# 它们在 shape 实例化时会自动执行 PyTorch 默认初始化，跳过即可。
_LAZY_MODULES = (nn.LazyConv2d, nn.LazyLinear)

def init_weights(module):
    """
    Xavier 均匀初始化

    对 Linear 和 Conv2d 层使用 Xavier Uniform 初始化
    """
    if isinstance(module, _LAZY_MODULES):
        return
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def init_kaiming(module):
    """
    Kaiming (He) 均匀初始化

    对 Linear 和 Conv2d 层使用 Kaiming Uniform 初始化，适合 ReLU 系列激活函数
    """
    if isinstance(module, _LAZY_MODULES):
        return
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def load_pretrained(model: nn.Module, pretrained_path: str, strict: bool = False, verbose: bool = True): 
    """
    加载预训练权重，自动处理维度不匹配
    
    Args:
        model: 目标模型
        pretrained_path: 权重文件路径
        strict: 是否严格匹配所有层
        verbose: 是否打印加载信息
    """
    if not pretrained_path:
        return model
    
    try:
        state_dict = torch.load(pretrained_path, map_location="cpu")
    except FileNotFoundError:
        print(f"⚠️ 预训练权重文件不存在: {pretrained_path}")
        return model
    
    # 处理不同保存格式
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]
    
    # 过滤维度不匹配的层
    model_state = model.state_dict()
    filtered_dict = {}
    skipped_keys = []
    
    for k, v in state_dict.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                filtered_dict[k] = v
            else:
                skipped_keys.append(f"{k} (shape mismatch: {v.shape} vs {model_state[k].shape})")
        else:
            skipped_keys.append(f"{k} (not in model)")
    
    # 加载权重
    model.load_state_dict(filtered_dict, strict=False)
    
    if verbose:
        print(f"✅ 加载预训练权重: {len(filtered_dict)}/{len(model_state)} 层匹配")
        if skipped_keys:
            print(f"   ⚠️ 跳过的层: {skipped_keys[:5]}{'...' if len(skipped_keys) > 5 else ''}")

    return model


# ============================================================================
# 迁移学习微调模式：仅特征提取 / 全量微调
# ============================================================================

FINETUNE_MODES = ('feature_extract', 'full')


def _get_backbone_root(model: nn.Module) -> nn.Module:
    """找到承载网络结构的根模块。

    torchvision 包装类（ResNet18/VGG16）结构在 self.model；AlexNet/MLP 在 self.net；
    裸模型则用自身。
    """
    for attr in ('model', 'net'):
        inner = getattr(model, attr, None)
        if isinstance(inner, nn.Module) and len(list(inner.children())) > 1:
            return inner
    return model


def apply_finetune_mode(model: nn.Module, mode: str = 'full') -> dict:
    """按微调模式设置参数 requires_grad，返回统计信息。

      full            全量微调：所有参数可训练（默认）
      feature_extract 仅特征提取：冻结整个骨干，只训练分类头（最后一个子模块）

    附带处理：冻结的 BatchNorm 强制保持 eval 模式——否则 train_epoch 里的
    model.train() 会把 BN 切回训练态，running stats 被当前数据更新，
    破坏"骨干当固定特征提取器"的语义。
    """
    if mode not in FINETUNE_MODES:
        raise ValueError(f"未知 finetune_mode: {mode}，可选 {FINETUNE_MODES}")

    if mode == 'full':
        for p in model.parameters():
            p.requires_grad = True
    else:
        root = _get_backbone_root(model)
        children = list(root.named_children())
        if not children:
            raise ValueError("模型没有可识别的子模块结构，无法应用冻结")

        # feature_extract：全部冻结，仅解冻最后一个子模块（分类头）
        for p in model.parameters():
            p.requires_grad = False
        for p in children[-1][1].parameters():
            p.requires_grad = True

        # 冻结的 BN 强制 eval（通过包装 model.train 实现，Trainer 每轮调用时生效）
        frozen_bns = [m for m in model.modules()
                      if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
                      and not any(p.requires_grad for p in m.parameters())]
        if frozen_bns:
            _orig_train = model.train

            def _train_keep_bn_eval(m=model, orig=_orig_train, bns=frozen_bns):
                def _wrapped(mode=True):
                    orig(mode)
                    for bn in bns:
                        bn.eval()
                    return m
                return _wrapped

            model.train = _train_keep_bn_eval()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'mode': mode,
        'trainable_params': trainable,
        'frozen_params': total - trainable,
        'trainable_ratio': trainable / total if total else 0.0,
    }


# ============================================================================
# 第三部分：模型定义
# ============================================================================

@register_model("mlp")
class MLP(nn.Module):
    """
    多层感知机（MLP）
    
    适用于 MNIST、Fashion-MNIST 等简单数据集
    """
    def __init__(self, num_classes: int = 10, hidden_dims: list = [128, 64], 
                 input_dim: int = 784, dropout: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.net = nn.Sequential(*layers)
        self.apply(init_kaiming)
    
    def forward(self, x):
        # 展平输入（从 [batch, 1, 28, 28] -> [batch, 784]）
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


@register_model("alexnet")
class AlexNet(nn.Module):
    """
    AlexNet
    
    适用于 CIFAR-10、CIFAR-100 等中等尺寸图像数据集
    """
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        
        self.net = nn.Sequential(
            # 卷积层
            nn.LazyConv2d(96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.LazyConv2d(256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.LazyConv2d(384, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.LazyConv2d(384, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.LazyConv2d(256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 全连接层
            nn.Flatten(),
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            nn.LazyLinear(num_classes)
        )
        self.apply(init_kaiming)
    
    def forward(self, x):
        return self.net(x)
    
    def get_feature_extractor(self):
        """获取特征提取器（去掉最后的分类层）"""
        return nn.Sequential(*list(self.net.children())[:-1])


@register_model("resnet18")
class ResNet18(nn.Module):
    """
    ResNet-18
    
    适用于 CIFAR-10、CIFAR-100、ImageNet 等数据集
    """
    def __init__(self, num_classes: int = 1000, pretrained: bool = False):
        super().__init__()
        
        # 使用 torchvision 的 ResNet18
        try:
            from torchvision.models import resnet18,ResNet18_Weights
            if pretrained:
                weights = ResNet18_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.model = resnet18(weights=weights)
            
            # 替换分类层
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
            
        except ImportError:
            print("⚠️ torchvision 未安装")
    
    def forward(self, x):
        return self.model(x) if hasattr(self, 'model') else self.model(x)
    
    def get_feature_dim(self):
        """获取特征维度"""
        try:
            return self.model.fc.in_features
        except AttributeError:
            return 512


@register_model("vgg16")
class VGG16(nn.Module):
    """
    VGG-16
    
    适用于 CIFAR-10、CIFAR-100、ImageNet 等数据集
    """
    def __init__(self, num_classes: int = 1000, pretrained: bool = False):
        super().__init__()
        
        try:
            from torchvision.models import vgg16
            self.model = vgg16(pretrained=pretrained)
            
            # 替换分类层
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, num_classes)
            
        except ImportError:
            print("⚠️ torchvision 未安装，请安装: pip install torchvision")
            raise
    
    def forward(self, x):
        return self.model(x)
    
    def get_feature_extractor(self):
        """获取特征提取器"""
        return nn.Sequential(*list(self.model.features.children()))

# ============================================================================
# 第四部分：便捷工具函数
# ============================================================================

def get_model_params(model: nn.Module) -> dict:
    """
    获取模型的参数统计信息
    
    Returns:
        dict: 包含总参数数、可训练参数数等信息
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "total_params_m": total_params / 1e6,  # 百万
        "trainable_params_m": trainable_params / 1e6,
    }

def print_model_summary(model: nn.Module, input_size: Optional[tuple] = None):
    """
    打印模型摘要信息
    
    Args:
        model: 模型实例
        input_size: 输入尺寸，用于显示每层输出形状
    """
    print("=" * 60)
    print(f"模型: {model.__class__.__name__}")
    print("-" * 60)
    
    # 参数统计
    params = get_model_params(model)
    print(f"总参数量: {params['total_params']:,} ({params['total_params_m']:.2f}M)")
    print(f"可训练参数: {params['trainable_params']:,} ({params['trainable_params_m']:.2f}M)")
    print(f"冻结参数: {params['frozen_params']:,}")
    
    # 打印结构（可选）
    if input_size:
        print("-" * 60)
        print("模型结构:")
        print(model)
    
    print("=" * 60)