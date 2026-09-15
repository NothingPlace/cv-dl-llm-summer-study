from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ============================================================
# 配置驱动的通用数据层（新增，不影响上面 FashionMNISTData 的旧用法）
#
# 路径 A：torchvision 标准数据集（FashionMNIST/MNIST/CIFAR10/CIFAR100/SVHN）
#         cfg 只需 name + root，自动 download，按 train/split 取训练测试集
# 路径 B：ImageFolder 自收集分类数据
#         cfg 需 name: image_folder + train_dir + val_dir，类别名=子目录名
# ============================================================

# 数据增强策略注册表：仅作用于训练集；评估集始终用确定性 transform
# none=不增强；light=轻量(随机裁剪+翻转)；standard=标准(加颜色抖动+旋转)
# 每个策略是 size -> List[transform] 的函数（size 在 build_transform 时才知道）
def _aug_none(size):
    return []

def _aug_light(size):
    return [
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(),
    ]

def _aug_standard(size):
    return [
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomRotation(15),
    ]

AUGMENTATION_POLICIES = {
    'none': _aug_none,
    'light': _aug_light,
    'standard': _aug_standard,
}


def build_transform(resize, rgb_expand=False, mean=(0.5,), std=(0.5,),
                    train=False, augmentation='none'):
    """统一的图像预处理 transform。

    train=True 且 augmentation != 'none' 时，训练集在 Resize 位置插入
    PIL 级数据增强（随机裁剪/翻转/颜色抖动/旋转）；评估集永不增强。
    rgb_expand=True 且 mean/std 长度为 1 时自动广播到 3 通道（适配预训练模型）。
    """
    mean = tuple(mean) * 3 if (rgb_expand and len(mean) == 1) else tuple(mean)
    std = tuple(std) * 3 if (rgb_expand and len(std) == 1) else tuple(std)
    size = tuple(resize)

    if train and augmentation not in ('none', None):
        if augmentation not in AUGMENTATION_POLICIES:
            raise ValueError(f"未知 augmentation: {augmentation}，"
                             f"可选 {list(AUGMENTATION_POLICIES)}")
        t = AUGMENTATION_POLICIES[augmentation](size)
    else:
        t = [transforms.Resize(size)]

    if rgb_expand:
        t.append(transforms.Grayscale(num_output_channels=3))
    t += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(t)


# 路径 A：name → torchvision Dataset 类
TORCHVISION_DATASETS = {
    'fashion_mnist': datasets.FashionMNIST,
    'mnist':         datasets.MNIST,
    'cifar10':       datasets.CIFAR10,
    'cifar100':      datasets.CIFAR100,
    'svhn':          datasets.SVHN,
    'flowers102':    datasets.Flowers102,
}

# 路径 B：ImageFolder
PATH_B_IMAGEFOLDER = {'image_folder'}

# 元信息：通道数 + 类别名兜底（dataset 自带 .classes 时优先用自带的）
DATASET_META = {
    'fashion_mnist': {'channels': 1, 
                      'classes': ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']},
    'mnist':         {'channels': 1, 'classes': [str(i) for i in range(10)]},
    'cifar10':       {'channels': 3,
                      'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                                  'dog', 'frog', 'horse', 'ship', 'truck']},
    'cifar100':      {'channels': 3, 'classes': []},   # 100 类，直接用 dataset.classes
    'svhn':          {'channels': 3, 'classes': [str(i) for i in range(10)]},
}


class ImageClassificationData:
    """统一图像分类数据管理器，对外接口与 FashionMNISTData 一致：
    get_train_loader / get_test_loader / text_labels / dataset_Resize。
    """

    def __init__(self, name='fashion_mnist', root='./data', resize=(224, 224),
                 rgb_expand=False,batch_size=128, mean=(0.5,), std=(0.5,), num_workers=0,
                 pin_memory=False, train_dir=None, val_dir=None, download=True,
                 augmentation='none'):
        self.name = name
        self.root = root
        self.resize = tuple(resize)
        self.rgb_expand = rgb_expand
        self.batch_size=batch_size
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.download = download
        self.augmentation = augmentation

        self._load_datasets()

    # ----------------------------------------------------------
    def _build_transform(self, train=False):
        """train=True 时带数据增强；评估集始终确定性 transform。"""
        return build_transform(self.resize, self.rgb_expand, self.mean, self.std,
                               train=train, augmentation=self.augmentation)

    def _load_datasets(self):
        """按 name 分派构建 train/test 数据集。"""
        train_transform = self._build_transform(train=True)
        eval_transform = self._build_transform(train=False)

        # --- 路径 B：ImageFolder ---
        if self.name in PATH_B_IMAGEFOLDER:
            self.train_dataset = datasets.ImageFolder(self.train_dir, transform=train_transform)
            self.test_dataset = datasets.ImageFolder(self.val_dir, transform=eval_transform)
            self.class_names = self.train_dataset.classes   # 类别名=子目录名
            return

        # --- 路径 A：torchvision 标准数据集 ---
        cls = TORCHVISION_DATASETS[self.name]
        if self.name in ('svhn', 'flowers102'):
            # SVHN 用 split 而非 train flag
            self.train_dataset = cls(root=self.root, split='train',
                                     download=self.download, transform=train_transform)
            self.test_dataset = cls(root=self.root, split='test',
                                    download=self.download, transform=eval_transform)
        else:
            self.train_dataset = cls(root=self.root, train=True,
                                     download=self.download, transform=train_transform)
            self.test_dataset = cls(root=self.root, train=False,
                                    download=self.download, transform=eval_transform)
        # 类别名：优先 dataset 自带 .classes，回退到元信息
        self.class_names = (getattr(self.train_dataset, 'classes', None)
                            or DATASET_META[self.name]['classes'])

    # ----------------------------------------------------------
    def get_train_loader(self, shuffle=True):
        return DataLoader(
            self.train_dataset,shuffle=shuffle,batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.pin_memory)

    def get_test_loader(self,shuffle=False):
        return DataLoader(
            self.test_dataset,shuffle=shuffle,batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.pin_memory)

    def text_labels(self, indices):
        """数字索引 → 文本标签。"""
        return [self.class_names[int(i)] for i in indices]


# ============================================================
# 工厂函数：config['data'] 子节 → 数据管理器
# ============================================================
def build_data(cfg: dict):
    """根据配置构建数据管理器。

    路径 A（cfg.name ∈ TORCHVISION_DATASETS）示例:
        data:
          name: fashion_mnist      # 或 mnist / cifar10 / cifar100 / svhn
          root: ./data
          resize: [96, 96]
          rgb: true
          mean: [0.5]
          std: [0.5]
          batch_size: 128
          num_workers: 2
          pin_memory: true

    路径 B（cfg.name == 'image_folder'）示例:
        data:
          name: image_folder
          train_dir: ./data/mydata/train
          val_dir: ./data/mydata/val
          resize: [224, 224]
          rgb_expand: true
    """
    data_cfg = cfg.get("data", {})
    name = data_cfg.get('name')
    common = dict(
        root=data_cfg.get('root', './data'),
        resize=tuple(data_cfg.get('resize', (224, 224))),
        rgb_expand=data_cfg.get('rgb_expand', False),
        batch_size=data_cfg.get('batch_size', 128),
        mean=tuple(data_cfg.get('mean', (0.5,))),
        std=tuple(data_cfg.get('std', (0.5,))),
        num_workers=data_cfg.get('num_workers', 0),
        pin_memory=data_cfg.get('pin_memory', False),
        download=data_cfg.get('download', True),
        augmentation=data_cfg.get('augmentation', 'none'),
    )
    if name in PATH_B_IMAGEFOLDER:
        return ImageClassificationData(
            name=name,
            train_dir=data_cfg.get('train_dir'),
            val_dir=data_cfg.get('val_dir'),
            **common)
    return ImageClassificationData(name=name, **common)
