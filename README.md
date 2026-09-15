# cv-dl-llm-summer-study

暑期学习记录

## 📋 目录

- [环境配置](#环境配置)
- [第一阶段：图像处理基础学习](#第一阶段图像处理基础学习)
- [第二阶段：深度学习与卷积神经网络](#第二阶段深度学习与卷积神经网络)
- [项目结构](#项目结构)

---

## 环境配置

本项目使用 Conda 管理虚拟环境，所有依赖已锁定在 `environment.yml` 中。

### 创建并激活环境

```bash
#  克隆仓库
git clone https://github.com/NothingPlace/cv-dl-llm-summer-study.git
cd cv-dl-llm-summer-study

# 从环境文件创建
conda env create -f environment.yml

# 激活环境（环境名称在 environment.yml 中定义）
conda activate d2l
```

## 第一阶段：图像处理基础学习

### Week01总结

#### Week01核心目标

- [ ] **目标一**：学习了解Python 与图像基础
- [ ] **目标二**：完成代码实践week01_image_io.py；

#### Week01阶段成果

- **一**：学习了解Python 与图像基础（详细笔记请查看：[week01.md](notes/week01.md) ）
- **二**：完成代码实践week01_image_io.py,实现读取图片；保存灰度图、HSV 图、缩放图；输出尺寸、通道数、像素值范围。（详细代码请查看：[week01_image_io.py](./image_processing/week01/week01_image_io.py) ）

#### week01_image_io.py 运行方式

```python
conda activate d2l #激活环境
cd cv-dl-llm-summer-study\image_processing\week01 #进入文件目录
python week01_image_io.py
```

### Week02总结

#### Week02核心目标

- [ ] **目标一**：学习了解增强、边缘等形态学处理技术
- [ ] **目标二**：完成代码实践week02_cv_demo.py

#### Week02阶段成果

- **一**：学习了解增强、边缘等形态学处理技术（详细笔记请查看：[week02.md](notes/week02.md) ）
- **二**：完成代码实践week02_cv_demo.py,实现字符区域定位。（详细代码请查看：[week02_cv_demo.py](./image_processing/week02/week02_cv_demo.py) ）

#### Week02运行方式

```python
conda activate d2l #激活环境
cd cv-dl-llm-summer-study\image_processing\week02 #进入文件目录
python week02_cv_demo.py
```

## 第二阶段：深度学习与卷积神经网络

### Week03总结

#### Week03核心目标

- [ ] **目标一**：学习了解深度学习训练流程
- [ ] **目标二**：完成代码实践week03_dl_basics.py

#### Week03阶段成果

- **一**：学习了解深度学习训练流程（详细笔记请查看：[week03.md](notes/week03.md) ）
- **二**：完成代码实践week03_dl_basics.py,实现softmax 分类。（详细代码请查看：[week03_dl_basics.py](./deep_learning/week03_dl_basics.py) ）

#### Week03运行方式

```python
conda activate d2l #激活环境
cd cv-dl-llm-summer-study\deep_learning #进入文件目录
python week03_dl_basics.py
```

### Week04总结

#### Week04核心目标

- [ ] **目标一**：学习了解多层感知机，dropout等相关内容
- [ ] **目标二**：完成代码实践week04_mlp_classification.py

#### Week04阶段成果

- **一**：学习了解多层感知机，dropout等相关内容（详细笔记请查看：[week04.md](notes/week04.md) ）
- **二**：完成代码实践week04_mlp_classification.py,实现MLP 分类。（详细代码请查看：[week04_mlp_classification.py](./deep_learning/week04_mlp_classification.py) ）

#### Week04运行方式

```python
conda activate d2l #激活环境
cd cv-dl-llm-summer-study\deep_learning #进入文件目录
python week04_mlp_classification.py
```

### Week05总结

#### Week05核心目标

- [ ] **目标一**：学习了解LeNet、简单 CNN等相关内容
- [ ] **目标二**：完成代码实践week05_cnn_baseline.py

#### Week05阶段成果

- **一**：学习了解LeNet、简单 CNN等相关内容（详细笔记请查看：[week05.md](notes/week05.md) ）
- **二**：完成代码实践week05_cnn_baseline.py,实现LeNet 分类。（详细代码请查看：[week05_cnn_baseline.py](./deep_learning/week05_cnn_baseline.py) ）

#### Week05运行方式

```python
conda activate d2l #激活环境
cd cv-dl-llm-summer-study\deep_learning #进入文件目录
python week05_cnn_baseline.py
```

### Week06总结

#### Week06核心目标

- [ ] **目标一**：学习了解AlexNet、VGG、GoogLeNet、ResNet等相关内容
- [ ] **目标二**：完成代码实践week06_transfer_learning.py

#### Week06阶段成果

- **一**：学习了解AlexNet、VGG、GoogLeNet、ResNet等相关内容（详细笔记请查看：[week06.md](notes/week06.md) ）
- **二**：完成代码实践week06_transfer_learning.py,实现LeNet 分类。（详细代码请查看：[week06_transfer_learning.py](./deep_learning/week06_transfer_learning.py) ）

注：由于AlexNet、VGG、GoogLeNet训练时间较长，相关训练图像放置在week06.md中。

#### Week06运行方式

```python
conda activate d2l #激活环境
cd cv-dl-llm-summer-study\deep_learning #进入文件目录
python week06_transfer_learning.py
```

### Week07总结

#### Week07核心目标

- [ ] **目标一**：学习了解数据增强、配置文件、checkpoint等相关内容
- [ ] **目标二**：完成代码实践train/test/dataset/model.py

#### Week07阶段成果

- **一**：学习了解数据增强、配置文件、checkpoint等相关内容（详细笔记请查看：[week07.md](notes/week07.md) ）
- **二**：完成代码实践train/test/dataset/model.py,实现通用训练代码。（详细代码请查看：[week07_pytorch_project.py](./deep_learning/week07_pytorch_project) ）

### Week08总结

#### Week08核心目标

- [ ] **目标一**：学习了解了解图像分类、目标检测、语义分割、ViT/Attention 的基本差异等相关内容
- [ ] **目标二**：确定最终项目候选方向

#### Week08阶段成果

- **一**：学习了解了解图像分类、目标检测、语义分割、ViT/Attention 的基本差异等相关内容（详细笔记请查看：[week08.md](notes/week08.md) ）
- **二**：确定最终项目候选方向，实现基于ResNet的花卉识别

### Week09总结

#### Week09核心目标

- [ ] **目标一**：确定最终视觉项目题目、数据集、类别、评价指标；完成数据集说明和初步预处理。
- [ ] **目标二**：完成代码实践llm_api_demo.py

#### Week09阶段成果

- **一**：学习了解AlexNet、VGG、GoogLeNet、ResNet等相关内容（详细笔记请查看：[week09.md](notes/week09.md) ）
- **二**：完成代码实践week06_transfer_learning.py,实现LeNet 分类。（详细代码请查看：[llm_api_demo.py](./llm_intro/api_demo/llm_api_demo.py) ）

#### Week09运行方式

```python
conda activate d2l #激活环境
cd cv-dl-llm-summer-study\llm_intro\api_demo #进入文件目录
python llm_api_demo.py
```

### Week10总结

#### Week10核心目标

- [ ] **目标一**：搭建项目 baseline：model.py、train.py、test.py；
- [ ] **目标二**：学习了解了解本地模型、量化、显存占用、推理速度、CPU/GPU 推理差异等相关内容
- [ ] **目标三**：完成本地部署/本地推理：任选一种工具路线，记录环境、模型、启动命令和推理截图。

#### Week10阶段成果

- **一**：学习了解了解本地模型、量化、显存占用、推理速度、CPU/GPU 推理差异等相关内容（详细笔记请查看：[week10.md](notes/week10.md) ）
- **二**：搭建项目 baseline：model.py、train.py、test.py。（详细代码请查看：[cv_project](./cv_project/) ）
- [ ] **三**：完成本地部署/本地推理（详细代码请查看：[local_deploy.py](./llm_intro/local_deploy/local_deploy.py) ）

#### Week10运行方式

```python
conda activate d2l #激活环境
cd cv-dl-llm-summer-study\llm_intro\local_deploy #进入文件目录
python local_deploy.py
cd cv-dl-llm-summer-study\cv_project #进入文件目录
python train.py
python test.py
```

### Week11总结

#### Week11核心目标

- [ ] **目标一**：学习了解学习消融实验、参数对比、数据增强对比、类别级准确率、混淆矩阵、错误样例分析等相关内容
- [ ] **目标二**：完成代码实践,实现不同模型参数的对比及可视化分析

#### Week11阶段成果

- **一**：学习了解学习消融实验、参数对比、数据增强对比、类别级准确率、混淆矩阵、错误样例分析等相关内容（详细笔记请查看：[week11.md](notes/week11.md) ）
- **二**：完成代码实践,实现不同模型参数的对比及可视化分析。（详细代码请查看：[cv_project](./cv_project/) ）

#### Week11运行方式

```python
conda activate d2l #激活环境
cd cv-dl-llm-summer-study\cv_project #进入文件目录
python experiments.py
```

### Week12总结

#### Week12核心目标

- [ ] **目标一**：完成12周学习总结
- [ ] **目标二**：完成项目报告ppt

#### Week12阶段成果

- **一**：完成12周学习总结（详细笔记请查看：[week12.md](notes/week12.md) ）
- **二**：完成代码实践week06_transfer_learning.py,实现LeNet 分类。（详细代码请查看：[week06_transfer_learning.py](./deep_learning/week06_transfer_learning.py) ）

注：由于AlexNet、VGG、GoogLeNet训练时间较长，相关训练图像放置在week06.md中。

## 项目结构

```项目结构
cv-dl-llm-summer-study/
├── README.md                         # 总说明：学习目标、环境、运行方式、阶段成果
├── requirements.txt 或 environment.yml # 依赖环境，不包含 API Key
├── notes/                            # 12 份周学习笔记
│   ├── week01.md
│   └── ...
├── image_processing/                 # 第 1-2 周图像处理代码
├── deep_learning/                    # 第 3-7 周深度学习代码
│   ├── week03_dl_basics.py
│   ├── week04_mlp_classification.py
│   ├── week05_cnn_baseline.py
│   ├── week06_transfer_learning.py
│   └── training_template/
├── llm_intro/                        # 第 8-10 周大模型入门必做内容
│   ├── llm_basics_notes.md
│   ├── api_demo/
│   │   ├── llm_api_demo.py
│   │   ├── prompt_cases.md
│   │   └── .env.example
│   └── local_deploy/
│       ├── local_deploy_notes.md
│       └── screenshots/
├── cv_project/                       # 第 9-12 周综合视觉项目
│   ├── data/README.md
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── test.py
│   ├── configs/
│   └── README.md
├── optional_extensions/              # 后续拓展：RAG、Agent、LoRA 等
├── results/                          # 结果图、训练曲线、实验表格、错误样例
└── report/                           # 总结报告或 PPT

```
