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

- **一**：学习了解增强、边缘等形态学处理技术（详细笔记请查看：[week03.md](notes/week03.md) ）
- **二**：完成代码实践week03_dl_basics.py,实现softmax 分类。（详细代码请查看：[week03_dl_basics.py](./deep_learning/week03_dl_basics.py) ）

#### Week03运行方式

```python
conda activate d2l #激活环境
cd cv-dl-llm-summer-study\deep_learning #进入文件目录
python week03_dl_basics.py
```

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