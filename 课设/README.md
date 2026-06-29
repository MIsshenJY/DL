# 基于知识蒸馏的轻量级中文情感分析模型

> 深度学习课程设计 —— 将 BERT 的知识蒸馏到 BiLSTM，实现轻量级中文情感分类

---

## 📌 项目简介

本项目是《深度学习B》课程设计作品。针对 BERT 等预训练语言模型参数量大、推理速度慢、难以部署在资源受限设备（如移动端、嵌入式设备）的问题，采用**知识蒸馏（Knowledge Distillation）**技术，将微调后的 BERT-base-chinese（教师模型）的知识迁移至轻量级 BiLSTM（学生模型），在保持较高准确率的同时大幅降低模型体积和推理时间。

实验在 **ChnSentiCorp** 中文情感分类数据集（酒店、数码、书籍等多领域评论）上进行，蒸馏后的学生模型在测试集上达到了 **91.25%** 的准确率，相比纯监督基线（89.50%）提升了 **1.75 个百分点**，同时参数量仅为教师模型的 **8.86%**（约 9M vs 102M）。

---

## 📊 实验结果

| 模型 | 参数量 | 测试准确率 | 测试 F1 |
|:---|:---:|:---:|:---:|
| 教师 BERT（微调） | 102.27M | **94.58%** | **94.58%** |
| BiLSTM（纯监督基线） | 9.06M | 89.50% | 89.50% |
| BiLSTM（知识蒸馏） | 9.06M | **91.25%** | **91.25%** |

> 蒸馏后的学生模型保留了教师模型约 **96.5%** 的分类性能，参数量压缩至教师的 **8.86%**。


---


## 🚀 快速开始

### 1. 环境配置

推荐使用 Python 3.9+，建议创建虚拟环境：

```bash
# 创建并激活虚拟环境（conda 示例）
conda create -n kd_sentiment python=3.9
conda activate kd_sentiment

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载必要资源

本项目需要以下预训练资源，请自行下载并放置到对应目录：

| 资源 | 来源 | 存放路径 |
|:---|:---|:---|
| BERT-base-chinese | Hugging Face | `models/bert-base-chinese-local/` |
| FastText 中文词向量 (300d) | FastText 官方 | `models/wiki.zh.vec` |
| ChnSentiCorp 数据集 | Hugging Face | `ChnSentiCorp/` |

> 如果网络条件允许，代码会自动从 Hugging Face 下载；否则请手动下载并放置到上述路径。

### 3. 运行代码

启动 Jupyter Notebook，按顺序运行所有单元格：

```bash
jupyter notebook "DL课程设计.ipynb"
```

> 建议先运行 `Kernel -> Restart & Run All` 确保所有输出正确。

---

## 🧠 核心方法

### 教师模型
- **模型**：BERT-base-chinese（12 层 Transformer）
- **参数量**：102.27M
- **微调**：在 ChnSentiCorp 训练集上微调 3 个 epoch，验证准确率 94.58%

### 学生模型
- **模型**：2 层双向 LSTM（BiLSTM）
- **嵌入层**：300 维，使用 FastText 预训练词向量初始化（可微调）
- **参数量**：9.06M

### 知识蒸馏
- **温度参数 T**：1.0
- **软标签权重 α**：0.3
- **损失函数**：`α * KL(soft_student || soft_teacher) * T² + (1-α) * CrossEntropy(student, hard_label)`
- **训练策略**：早停（patience=3）+ 学习率衰减（ReduceLROnPlateau）

---

## 📈 结果分析

1. **蒸馏有效**：蒸馏后准确率（91.25%）比基线（89.50%）提升 1.75 个百分点，达到预期目标（≥1.5%）。
2. **压缩显著**：学生模型参数量（9.06M）仅为教师（102.27M）的 8.86%，推理速度预计提升 5~10 倍。
3. **泛化良好**：从混淆矩阵可见，模型在正负类别上分类错误分布均衡，未出现严重偏向。
