# AI文本检测系统

基于深度学习的AI文本检测系统，用于区分人类撰写的文本和人工智能生成的文本。

## 系统特点

- 支持多种机器学习和深度学习模型
- 灵活的数据加载和预处理模块
- 高效的特征提取方法
- 详细的模型评估和可视化功能
- 支持集成学习方法
- 友好的Web应用界面

## 数据集

本系统使用AI与人类生成文本数据集，位于`dataset/AI-and-Human-Generated-Text/`目录下，包含以下文件：

- `combined_ai_gen_dataset.csv`: 包含`abstract`和`label`列的数据集，其中`label`为0表示人类撰写，1表示AI生成
- `train.csv`: 训练集，包含`title`、`abstract`和`label`列
- `test.csv`: 测试集，格式与训练集相同

## 快速开始

### 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 数据集分析

首先分析数据集，了解数据分布情况：

```bash
python train.py --analyze_data
```

### 训练模型

使用优化后的训练脚本可以快速训练多种AI文本检测模型：

```bash
# 使用combined数据集和随机森林模型训练
python train.py --data_dir ./dataset --data_type combined --model_type random_forest --model_name rf_model

# 使用训练集和测试集分离的方式训练逻辑回归模型
python train.py --data_dir ./dataset --use_test_split --model_type logistic --model_name lr_model

# 使用BERT模型训练
python train.py --data_dir ./dataset --data_type train --model_type bert --pretrained_model bert-base-uncased --model_name bert_model --feature_type transformer
```

### 主要参数说明

#### 数据参数

- `--data_dir`: 数据集目录，默认为`./dataset`
- `--data_type`: 数据集类型，可选`combined`或`train`
- `--use_test_split`: 使用预定义的训练/测试集拆分（train.csv和test.csv）
- `--balanced`: 是否平衡数据集，默认True
- `--analyze_data`: 分析数据集结构和分布

#### 预处理参数

- `--feature_type`: 特征类型，可选`tfidf`, `count`, `transformer`, `combined`
- `--clean_method`: 文本清洗方法，可选`basic`, `advanced`, `none`
- `--max_features`: 最大特征数量
- `--max_length`: 最大序列长度

#### 模型参数

- `--model_type`: 模型类型，可选`logistic`, `random_forest`, `svm`, `gradient_boosting`, `bert`, `roberta`
- `--model_name`: 模型名称
- `--pretrained_model`: 预训练模型名称，默认为`bert-base-uncased`

### 使用训练好的模型

#### 检测单个文本

```bash
python detect.py --text "待检测的文本内容" --model_path ./models/rf_model.pkl --model_type random_forest
```

#### 检测文本文件

```bash
python detect.py --file sample.txt --model_path ./models/lr_model.pkl --model_type logistic --output result.json
```

#### 批量检测CSV文件

```bash
python detect.py --csv dataset/AI-and-Human-Generated-Text/test.csv --column abstract --model_path ./models/bert_model.pt --model_type bert --feature_type transformer --output batch_results.json
```

### 启动Web应用

```bash
python app.py --model_path ./models/rf_model.pkl --model_type random_forest
```

## 项目结构

```
AITextInspector/
├── app.py                      # Web应用启动脚本
├── train.py                    # 模型训练脚本
├── detect.py                   # 模型检测脚本
├── requirements.txt            # 项目依赖
├── README.md                   # 项目说明
├── dataset/                    # 数据目录
│   └── AI-and-Human-Generated-Text/
│       ├── combined_ai_gen_dataset.csv
│       ├── train.csv
│       └── test.csv
├── models/                     # 模型目录
└── src/                        # 源代码
    ├── __init__.py
    ├── app/                    # Web应用模块
    ├── data_processing/        # 数据处理模块
    │   ├── __init__.py
    │   ├── data_loader.py      # 数据加载
    │   └── preprocess.py       # 文本预处理
    ├── models/                 # 模型模块
    │   ├── __init__.py
    │   └── model_factory.py    # 模型工厂
    ├── evaluation/             # 评估模块
    │   ├── __init__.py
    │   └── metrics.py          # 评估指标
    └── utils/                  # 工具模块
        └── __init__.py
```

## 参考资料

- [AI-and-Human-Generated-Text Dataset](https://github.com/example/AI-and-Human-Generated-Text)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Scikit-learn](https://scikit-learn.org/)

## 许可证

本项目基于 [MIT](LICENSE) 许可证开源。

