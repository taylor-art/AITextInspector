#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于深度学习的AI文本检测系统训练脚本
主要基于BERT模型实现

BERT(Bidirectional Encoder Representations from Transformers)是一种预训练语言模型，
能够理解文本的上下文语义。相比传统机器学习方法，BERT能够更好地捕捉文本的深层语义特征，
因此特别适合AI生成文本检测这类需要深入理解文本的任务。
"""

import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import time
import platform
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import json

# 导入自定义模块
from src.data_processing.data_loader import DataLoader as CustomDataLoader  # 重命名以避免冲突
from src.data_processing.preprocess import TextPreprocessor
from src.models.model_factory import ModelFactory
from src.evaluation.metrics import calculate_metrics, plot_confusion_matrix

# 配置日志 - 用于记录训练过程中的信息，方便调试和查看进度
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("training.log"), 
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

# 定义轻量级模型映射
LIGHTWEIGHT_MODELS = {
    'bert': 'prajjwal1/bert-tiny',  # 4层微型BERT
    'roberta': 'distilroberta-base'  # 更小的RoBERTa变体
}

def check_environment():
    """
    检查运行环境，提供相应建议
    
    返回环境信息字典，包含Python版本、操作系统、GPU可用性和内存情况
    """
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"操作系统: {platform.system()} {platform.release()}")
    
    # 检查GPU可用性
    gpu_available = False
    try:
        gpu_available = torch.cuda.is_available()
        logger.info(f"PyTorch GPU可用: {gpu_available}")
        if not gpu_available:
            logger.warning("未检测到GPU，深度学习模型训练可能较慢")
    except ImportError:
        logger.warning("PyTorch未安装，无法检测GPU")
    
    # 检查内存情况
    try:
        import psutil
        mem = psutil.virtual_memory()
        logger.info(f"系统内存: 总计 {mem.total/1024**3:.1f}GB, 可用 {mem.available/1024**3:.1f}GB")
        memory_info = {
            "memory_total": mem.total,
            "memory_available": mem.available
        }
    except ImportError:
        logger.warning("psutil未安装，无法检测系统内存")
        memory_info = {}
    
    return {
        "python_version": sys.version,
        "os": f"{platform.system()} {platform.release()}",
        "gpu_available": gpu_available,
        **memory_info
    }

def incremental_train(model, data_loader, batch_size=32, epochs=1):
    """
    分批次训练，避免一次加载所有数据
    
    Args:
        model: 要训练的模型
        data_loader: 数据加载器
        batch_size: 批次大小
        epochs: 训练轮数
        
    Returns:
        训练后的模型
    """
    # 这里需要根据模型类型实现不同的增量训练逻辑
    # 以下是一个简化的示例
    logger.info("使用增量学习模式训练")
    
    total_batches = len(data_loader) // batch_size + 1
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        for batch_idx in range(total_batches):
            # 获取一批数据
            try:
                X_batch, y_batch = data_loader.get_batch(batch_idx, batch_size)
                logger.info(f"  训练批次 {batch_idx+1}/{total_batches}")
                
                # 训练模型
                model.partial_fit(X_batch, y_batch)
            except (StopIteration, IndexError):
                logger.info("  数据已处理完毕")
                break
    
    return model

def train(args):
    """
    模型训练主函数 - 整个AI文本检测模型训练的核心流程
    
    深度学习训练流程通常包括：数据准备、预处理、特征提取、模型构建、模型训练和评估。
    本函数按照这个流程组织，将各步骤模块化处理。
    
    Args:
        args: 命令行参数对象，包含各种训练配置
    """
    start_time = time.time()
    logger.info("开始训练...")
    logger.info(f"数据集目录: {args.data_dir}")
    logger.info(f"模型类型: {args.model_type}")
    
    # 应用CPU优化配置
    if args.cpu_optimize:
        logger.info("应用CPU优化配置...")
        args.max_length = min(args.max_length, 128)  # 减小序列长度
        args.batch_size = min(args.batch_size, 4)    # 减小批次大小
        if args.max_samples is None or args.max_samples > 1000:
            args.max_samples = 1000                   # 限制样本数量
        logger.info(f"优化后配置: max_length={args.max_length}, batch_size={args.batch_size}, max_samples={args.max_samples}")
    
    # 使用轻量级模型
    if args.use_lightweight_model and args.model_type in ['bert', 'roberta']:
        original_model = args.pretrained_model
        args.pretrained_model = LIGHTWEIGHT_MODELS.get(args.model_type, args.pretrained_model)
        logger.info(f"使用轻量级模型: {original_model} → {args.pretrained_model}")
    
    # 1. 数据分析 - 了解数据分布情况，有助于选择合适的模型和参数
    if args.analyze_data:
        logger.info("分析数据集...")
        data_loader = CustomDataLoader(args.data_dir, max_samples=args.max_samples)
        stats = data_loader.get_dataset_stats()
        for data_type, data_stats in stats.items():
            logger.info(f"数据集 {data_type} 统计信息:")
            for key, value in data_stats.items():
                logger.info(f"  {key}: {value}")
    
    # 2. 数据加载 - 从磁盘读取数据并准备训练
    logger.info("加载数据...")
    data_loader = CustomDataLoader(args.data_dir, max_samples=args.max_samples)
    
    # 准备训练和测试数据
    # 深度学习需要将数据分为训练集和测试集，训练集用于模型学习，测试集用于评估模型性能
    if args.use_test_split:
        # 使用预定义的训练/测试集拆分 - 适用于数据集已经划分好的情况
        logger.info("使用预定义的训练/测试集拆分...")
        X_train, X_test, y_train, y_test = data_loader.prepare_data(
            data_type='test_split',
            balanced=args.balanced  # balanced参数控制是否平衡正负样本数量，防止样本不平衡导致模型偏倚
        )
    else:
        # 自动拆分数据集 - 从同一个数据集中划分训练集和测试集
        logger.info(f"使用数据集 {args.data_type} 并拆分为训练/测试集...")
        X_train, X_test, y_train, y_test = data_loader.prepare_data(
            data_type=args.data_type,
            test_size=args.test_size,  # test_size控制测试集比例，通常为20%-30%
            random_state=args.random_state,  # random_state确保实验可重复性
            balanced=args.balanced
        )
    
    # 3. 文本预处理 - 将原始文本转换为机器学习算法可处理的特征
    logger.info("预处理文本...")
    # 默认使用深度学习预训练模型进行特征提取
    # BERT等预训练模型需要特定的tokenizer(分词器)来处理文本，使其符合模型输入格式
    preprocessor = TextPreprocessor(
        max_features=args.max_features,  # 最大特征数量，控制词汇表大小
        max_length=args.max_length,  # 最大序列长度，BERT通常限制为512个token
        transformer_model=args.pretrained_model,  # 使用指定的预训练模型
        use_multiprocessing=args.use_multiprocessing  # 多进程加速处理
    )
    
    # 生成特征 - 对于BERT模型默认使用transformer特征
    # BERT需要特殊的特征表示，与传统的TF-IDF或词袋模型不同
    feature_type = args.feature_type
    if args.model_type in ['bert', 'roberta'] and feature_type != 'transformer':
        # BERT模型最好使用transformer特征，否则会降低性能
        logger.warning(f"对于{args.model_type}模型，推荐使用transformer特征类型。当前设置为{feature_type}")
        if args.force_feature_type:
            # 用户强制使用指定特征类型
            logger.info(f"保持使用{feature_type}特征类型")
        else:
            # 自动切换到最适合BERT的特征类型
            logger.info("自动切换到transformer特征类型")
            feature_type = 'transformer'
    
    # 特征提取 - 将文本转换为数值特征
    # 对于BERT，特征是文本的token ID和注意力掩码
    # 对于传统机器学习，特征可能是TF-IDF向量或词频统计
    X_train_features, X_test_features = preprocessor.generate_features(
        X_train, X_test, 
        feature_type=feature_type,
        clean_method=args.clean_method  # 文本清洗方法，去除噪声
    )
    
    # 4. 创建和训练模型 - 核心步骤，构建AI文本检测模型并训练
    logger.info(f"创建{args.model_type}模型...")
    model_factory = ModelFactory(output_dir=args.output_dir)
    
    # 根据模型类型准备参数 - 不同模型需要不同的参数设置
    model_params = {}
    if args.model_type == 'logistic':
        # 逻辑回归是简单的线性分类器，适合作为基线模型
        model_params = {
            'C': args.c_value,  # C是正则化参数，控制模型复杂度
            'max_iter': args.max_iter  # 最大迭代次数，影响训练时间
        }
    elif args.model_type == 'random_forest':
        # 随机森林是集成学习方法，由多个决策树组成
        model_params = {
            'n_estimators': args.n_estimators,  # 决策树数量，通常越多越好
            'max_depth': args.max_depth  # 树的最大深度，控制复杂度
        }
    elif args.model_type == 'svm':
        # 支持向量机，寻找最佳分割超平面的算法
        model_params = {
            'C': args.c_value,  # 正则化参数
            'kernel': args.kernel  # 核函数，决定数据映射方式
        }
    elif args.model_type == 'gradient_boosting':
        # 梯度提升，通过迭代优化损失函数的集成方法
        model_params = {
            'n_estimators': args.n_estimators,  # 基学习器数量
            'learning_rate': args.learning_rate,  # 学习率控制每步更新幅度
            'max_depth': args.max_depth  # 树的最大深度
        }
    elif args.model_type in ['bert', 'roberta']:
        # BERT/RoBERTa是Transformer架构的预训练语言模型
        # 它们能通过自注意力机制捕捉文本中的长距离依赖关系
        model_params = {
            'model_name': args.pretrained_model,  # 预训练模型名称
            'num_labels': 2,  # 二分类任务：AI生成(1)或人类撰写(0)
            'dropout_rate': args.dropout_rate  # Dropout是防止过拟合的技术
        }
    
    # 创建模型 - 工厂模式，根据类型创建不同模型
    model = model_factory.create_model(args.model_type, **model_params)
    
    # 使用优化后的训练器
    trainer = EnhancedAITextDetectorTrainer(model, output_dir="./output")

    # 设置改进的训练参数
    trainer.setup_training(
        learning_rate=5e-5,  # 增加学习率
        weight_decay=0.005,  # 减小权重衰减
        early_stopping_patience=5,  # 增加早停耐心值
        scheduler_patience=3,  # 增加调度器耐心值
        gradient_clip_val=1.0  # 梯度裁剪阈值
    )

    # 开始训练，使用改进的参数
    history = trainer.train(
        train_dataset=X_train_features,
        val_dataset=X_test_features,
        batch_size=32,  # 增加批次大小
        epochs=None,  # 不限制训练轮数
        accumulation_steps=2,  # 梯度累积步数
        early_stopping=True
    )
    
    # 5. 模型评估 - 计算各种指标评估模型性能
    logger.info("计算评估指标...")
    # 评估指标包括准确率、精确率、召回率、F1值等，全面了解模型性能
    metrics = calculate_metrics(y_test, history['val_preds'], history['val_probas'] if args.model_type not in ['bert', 'roberta'] else None)
    
    for metric, value in metrics.items():
        # 检查value是否为标量，避免格式化NumPy数组
        if hasattr(value, 'shape') and len(getattr(value, 'shape', [])) > 0:
            logger.info(f"{metric}: {value}")
        else:
            try:
                logger.info(f"{metric}: {float(value):.4f}")
            except (TypeError, ValueError):
                logger.info(f"{metric}: {value}")
    
    # 打印详细的分类报告 - 包括每个类别的精确率、召回率、F1值
    logger.info("分类报告:")
    logger.info("\n" + classification_report(y_test, history['val_preds']))
    
    # 绘制混淆矩阵 - 可视化预测结果，直观了解模型分类效果
    # 混淆矩阵显示了预测正确和错误的样本数量
    plot_confusion_matrix(y_test, history['val_preds'], 
                          title=f'{args.model_type.capitalize()} Model Confusion Matrix',
                          save_path=os.path.join(args.output_dir, f"{args.model_name}_confusion_matrix.png"))
    
    # 6. 保存模型 - 将训练好的模型保存到磁盘，便于后续使用
    model_path = model_factory.save_model(model, args.model_type, args.model_name)
    logger.info(f"模型已保存到: {model_path}")
    
    end_time = time.time()
    logger.info(f"训练完成！总耗时: {(end_time - start_time)/60:.2f}分钟")
    
    return metrics, model_path

def parse_args(args_list=None):
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='基于深度学习的AI文本检测系统训练脚本')
    
    # 数据参数 - 控制数据加载和处理
    parser.add_argument('--data_dir', type=str, default='./dataset', 
                        help='数据集目录，存放训练数据的文件夹路径')
    parser.add_argument('--data_type', type=str, default='train', 
                        choices=['combined', 'train'],
                        help='数据集类型，选择使用哪个数据文件')
    parser.add_argument('--use_test_split', action='store_true',
                        help='使用预定义的训练/测试集拆分（train.csv和test.csv）')
    parser.add_argument('--test_size', type=float, default=0.2, 
                        help='测试集比例，范围0-1，通常为0.2或0.3')
    parser.add_argument('--random_state', type=int, default=42, 
                        help='随机种子，确保实验可重复性')
    parser.add_argument('--balanced', type=bool, default=True,
                        help='是否平衡数据集，确保正负样本数量相等')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大样本数量，用于测试或资源有限时')
    parser.add_argument('--analyze_data', action='store_true',
                        help='分析数据集，输出统计信息')
    
    # 预处理参数 - 控制文本处理和特征提取
    parser.add_argument('--max_features', type=int, default=10000, 
                        help='最大特征数量，控制词汇表大小，影响模型复杂度')
    parser.add_argument('--max_length', type=int, default=512, 
                        help='最大序列长度，BERT模型通常限制为512个token')
    parser.add_argument('--feature_type', type=str, default='transformer', 
                        choices=['tfidf', 'count', 'transformer', 'combined'],
                        help='特征类型：tfidf(词频-逆文档频率)、count(词频统计)、transformer(深度特征)、combined(组合特征)')
    parser.add_argument('--force_feature_type', action='store_true',
                        help='强制使用指定的特征类型，即使与模型类型不匹配')
    parser.add_argument('--clean_method', type=str, default='basic', 
                        choices=['basic', 'advanced', 'none'],
                        help='文本清洗方法：basic(基本清洗)、advanced(高级清洗)、none(不清洗)')
    parser.add_argument('--use_multiprocessing', action='store_true',
                        help='使用多进程加速预处理和训练（利用多核CPU）')
    
    # CPU优化参数 - 针对无GPU环境的优化
    parser.add_argument('--cpu_optimize', action='store_true',
                        help='针对CPU环境优化参数配置，自动减小批量大小和序列长度')
    parser.add_argument('--use_lightweight_model', action='store_true',
                        help='使用轻量级模型替代标准模型，减少训练时间')
    parser.add_argument('--incremental_learning', action='store_true',
                        help='使用增量学习，逐批处理数据，减少内存使用')
    
    # 早停参数 - 避免过度训练
    parser.add_argument('--early_stopping', action='store_true',
                        help='启用早停机制，在验证集性能不再提升时提前结束训练')
    parser.add_argument('--patience', type=int, default=2,
                        help='早停耐心值，连续多少轮验证集性能不提升则停止训练')
    
    # 模型参数 - 控制模型类型和核心设置
    parser.add_argument('--model_type', type=str, default='bert', 
                        choices=['logistic', 'random_forest', 'svm', 'gradient_boosting', 'bert', 'roberta'],
                        help='模型类型：logistic(逻辑回归)、random_forest(随机森林)、svm(支持向量机)、gradient_boosting(梯度提升)、bert(BERT模型)、roberta(RoBERTa模型)')
    parser.add_argument('--model_name', type=str, default='ai_text_detector',
                       help='模型名称，保存文件的前缀')
    parser.add_argument('--pretrained_model', type=str, default='desklib/ai-text-detector-v1.01',
                        help='预训练模型名称，BERT/RoBERTa的预训练模型')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout比率（仅用于深度学习模型），防止过拟合的重要参数')
                        
    # 逻辑回归参数 - 特定于逻辑回归模型
    parser.add_argument('--c_value', type=float, default=1.0,
                        help='正则化参数C，值越小正则化越强，防止过拟合')
    parser.add_argument('--max_iter', type=int, default=1000,
                        help='最大迭代次数，影响逻辑回归的训练时间')
                        
    # 随机森林参数 - 特定于随机森林模型
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='决策树数量，通常越多性能越好，但训练越慢')
    parser.add_argument('--max_depth', type=int, default=None,
                        help='最大深度，控制树的复杂度，防止过拟合')
                        
    # SVM参数 - 特定于支持向量机模型
    parser.add_argument('--kernel', type=str, default='linear',
                        choices=['linear', 'rbf', 'poly'],
                        help='核函数：linear(线性核)、rbf(径向基核)、poly(多项式核)')
                        
    # 梯度提升参数 - 特定于梯度提升模型
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='学习率，控制每步更新幅度，影响训练稳定性和速度')
                        
    # 深度学习参数 - 特定于BERT/RoBERTa等深度学习模型
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数,不指定则训练到收敛为止')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小，影响内存使用和训练速度，通常为8/16/32/64')
    parser.add_argument('--weight_decay', type=float, default=0.005,
                        help='权重衰减，用于正则化，防止过拟合')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='早停耐心值，连续多少轮验证集性能不提升则停止训练')
    parser.add_argument('--scheduler_patience', type=int, default=3,
                        help='学习率调度器耐心值，连续多少轮性能不提升则调整学习率')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0,
                        help='梯度裁剪阈值，用于防止梯度爆炸')
    parser.add_argument('--accumulation_steps', type=int, default=2,
                        help='梯度累积步数，用于模拟更大的批次大小')
    
    # 输出参数 - 控制结果保存
    parser.add_argument('--output_dir', type=str, default='./models', 
                        help='输出目录，保存模型和结果的位置')
    
    # 使用tensorboard
    parser.add_argument('--use_tensorboard', type=bool, default=True,
                        help='是否使用tensorboard记录训练过程')
    
    if args_list is not None:
        return parser.parse_args(args_list)
    return parser.parse_args()

# 脚本入口点
if __name__ == '__main__':
    # 检查环境
    env_info = check_environment()
    
    # 解析命令行参数
    args = parse_args()
    
    # 根据环境自动调整参数
    if not env_info.get("gpu_available", False) and not args.cpu_optimize:
        logger.warning("未检测到GPU，建议使用--cpu_optimize参数优化训练设置")
        logger.warning("推荐命令: python train.py --cpu_optimize --use_lightweight_model")
    
    # 确保输出目录存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 执行训练
    metrics, model_path = train(args)
    
    # 输出训练结果
    logger.info(f"模型训练完成，保存到：{model_path}")
    logger.info(f"模型性能：{metrics}")

class EnhancedAITextDetectorTrainer:
    """
    增强版AI文本检测模型训练器
    """
    def __init__(self, model, device=None, output_dir="./models"):
        """
        初始化训练器
        
        Args:
            model: AI文本检测模型
            device: 训练设备 ('cuda' or 'cpu')
            output_dir: 输出目录
        """
        self.model = model
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.log_dir = os.path.join(output_dir, 'logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # 初始化类别权重和预测阈值
        self.class_weights = None
        self.prediction_threshold = 0.5  # 初始阈值设为0.5
        self.best_threshold = 0.5  # 记录最佳阈值
        
        # 创建必要的目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        
        # 将模型移至设备
        self.model.to(self.device)
        self.training_config = {}

    def _calculate_class_weights(self, labels):
        """
        改进的类别权重计算方法
        使用逆频率权重并进行归一化
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        weights = np.array([1/c for c in counts])
        weights = weights / weights.sum() * len(weights)  # 归一化
        
        # 记录权重信息
        logger.info(f"计算的类别权重: 类别0={weights[0]:.4f}, 类别1={weights[1]:.4f}")
        
        return torch.tensor(weights, device=self.device, dtype=torch.float32)

    def _adjust_threshold(self, val_metrics):
        """
        动态调整预测阈值
        基于验证集的预测概率分布来优化阈值
        """
        if 'probabilities' not in val_metrics or 'labels' not in val_metrics:
            return

        probas = val_metrics['probabilities']
        labels = val_metrics['labels']
        
        if len(probas) > 0:
            # 使用ROC曲线找到最佳阈值
            from sklearn.metrics import roc_curve, f1_score
            fpr, tpr, thresholds = roc_curve(labels, probas)
            
            # 计算每个阈值下的F1分数
            f1_scores = []
            for threshold in thresholds:
                preds = (probas > threshold).astype(int)
                f1 = f1_score(labels, preds)
                f1_scores.append(f1)
            
            # 找到最佳F1分数对应的阈值
            best_idx = np.argmax(f1_scores)
            new_threshold = thresholds[best_idx]
            
            # 限制阈值在合理范围内
            new_threshold = max(0.2, min(0.8, new_threshold))
            
            if abs(new_threshold - self.prediction_threshold) > 0.05:
                old_threshold = self.prediction_threshold
                self.prediction_threshold = new_threshold
                logger.info(f"调整预测阈值: {old_threshold:.4f} -> {new_threshold:.4f}")
                
                # 记录到TensorBoard
                self.writer.add_scalar('threshold', new_threshold, self.current_epoch)

    def setup_training(self, learning_rate=5e-5, weight_decay=0.005, 
                      early_stopping_patience=5, scheduler_patience=3,
                      gradient_clip_val=1.0):
        """
        设置改进的训练参数
        """
        # 使用AdamW优化器，更适合Transformer模型
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999),  # 默认的Adam动量参数
            eps=1e-8  # 数值稳定性参数
        )
        
        # 改进的学习率调度器：使用余弦退火策略
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5,  # 第一次重启的周期
            T_mult=2,  # 每次重启后周期长度的倍数
            eta_min=1e-6  # 最小学习率
        )
        
        # 训练配置
        self.training_config.update({
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'early_stopping_patience': early_stopping_patience,
            'scheduler_patience': scheduler_patience,
            'gradient_clip_val': gradient_clip_val
        })
        
        logger.info(f"训练配置: {self.training_config}")

    def _create_dataloader(self, dataset, batch_size, shuffle=True):
        """
        创建数据加载器
        
        Args:
            dataset: 数据集（可以是字典或TensorDataset）
            batch_size: 批次大小
            shuffle: 是否打乱数据
            
        Returns:
            DataLoader: PyTorch数据加载器
        """
        if isinstance(dataset, dict):
            # 如果是字典格式，转换为TensorDataset
            input_ids = dataset['input_ids']
            attention_mask = dataset['attention_mask']
            
            # 确保是PyTorch张量
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)
            if not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.tensor(attention_mask)
            
            # 检查是否有标签
            if 'labels' in dataset:
                labels = dataset['labels']
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels, dtype=torch.long)
                tensor_dataset = TensorDataset(input_ids, attention_mask, labels)
            else:
                tensor_dataset = TensorDataset(input_ids, attention_mask)
            
            return DataLoader(
                tensor_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0  # 在CPU环境下使用0个worker
            )
        
        elif isinstance(dataset, TensorDataset):
            # 如果已经是TensorDataset，直接使用
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0
            )
        
        else:
            raise ValueError("不支持的数据集类型。必须是字典或TensorDataset。")

    def _process_batch(self, batch, training=True):
        """
        处理一个批次的数据
        """
        # 将批次数据移到设备
        batch = [b.to(self.device) for b in batch]
        
        # 检查批次数据的长度
        if len(batch) == 3:
            input_ids, attention_mask, labels = batch
            # 首次计算类别权重
            if training and self.class_weights is None:
                self.class_weights = self._calculate_class_weights(labels.cpu().numpy())
        elif len(batch) == 2:
            input_ids, attention_mask = batch
            labels = torch.zeros(input_ids.size(0), dtype=torch.long, device=self.device)
        else:
            raise ValueError(f"批次数据格式错误：期望2或3个张量，实际得到{len(batch)}个")
        
        # 前向传播
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs[0]
        logits = outputs[1]
        
        # 使用类别权重调整损失
        if training and self.class_weights is not None:
            criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = criterion(logits, labels)
        
        # 计算预测结果和概率
        probas = torch.softmax(logits, dim=1)[:, 1]  # 获取正类的概率
        preds = (probas > self.prediction_threshold).long()  # 使用阈值进行预测
        
        # 记录预测分布
        if not training:
            self.writer.add_histogram('predictions/probabilities', probas.cpu().numpy(), self.current_epoch)
            pred_dist = torch.bincount(preds, minlength=2)
            self.writer.add_text('predictions/distribution', 
                               f'Class 0: {pred_dist[0]}, Class 1: {pred_dist[1]}', 
                               self.current_epoch)
        
        if training:
            return loss
        else:
            return loss, preds, labels, probas

    def train(self, train_dataset, val_dataset=None, batch_size=32, epochs=None,
              accumulation_steps=2, early_stopping=True):
        """
        改进的训练方法,支持无限训练直到收敛
        """
        self.current_epoch = 0
        start_time = time.time()
        
        # 创建数据加载器
        train_dataloader = self._create_dataloader(train_dataset, batch_size, shuffle=True)
        val_dataloader = self._create_dataloader(val_dataset, batch_size) if val_dataset else None
        
        # 初始化训练历史
        history = {
            'train_loss': [], 'val_loss': [],
            'val_accuracy': [], 'val_precision': [],
            'val_recall': [], 'val_f1': [],
            'learning_rates': [],
            'val_preds': None,
            'val_probas': None,
            'thresholds': []
        }
        
        # 早停相关变量
        best_val_f1 = 0
        best_model_path = None
        no_improvement_count = 0
        max_epochs = epochs if epochs is not None else float('inf')  # 如果未指定epochs则设为无限
        
        # 训练循环
        while self.current_epoch < max_epochs:
            train_loss = self._train_epoch(train_dataloader, accumulation_steps)
            history['train_loss'].append(train_loss)
            history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            if val_dataloader:
                # 验证
                val_metrics = self._validate(val_dataloader)
                
                # 调整预测阈值
                self._adjust_threshold(val_metrics)
                history['thresholds'].append(self.prediction_threshold)
                
                # 记录验证指标
                history['val_loss'].append(val_metrics['loss'])
                history['val_accuracy'].append(val_metrics['accuracy'])
                history['val_precision'].append(val_metrics['precision'])
                history['val_recall'].append(val_metrics['recall'])
                history['val_f1'].append(val_metrics['f1'])
                history['val_preds'] = val_metrics['predictions']
                history['val_probas'] = val_metrics.get('probabilities', None)
                
                # 记录到TensorBoard
                self._log_metrics(self.current_epoch, train_loss, val_metrics)
                
                # 更新学习率
                self.scheduler.step(val_metrics['f1'])
                
                # 保存最佳模型
                if val_metrics['f1'] > best_val_f1:
                    best_val_f1 = val_metrics['f1']
                    no_improvement_count = 0
                    best_model_path = self._save_checkpoint(self.current_epoch, val_metrics['f1'])
                else:
                    no_improvement_count += 1
                
                # 早停检查
                if early_stopping and no_improvement_count >= self.training_config['early_stopping_patience']:
                    logger.info(f"早停：{self.training_config['early_stopping_patience']}轮未见改善")
                    break
            
            # 打印进度
            self._print_epoch_summary(self.current_epoch, "未限制" if epochs is None else epochs, 
                                    train_loss, val_metrics if val_dataloader else None)
            
            self.current_epoch += 1
        
        # 训练结束后的处理
        training_time = time.time() - start_time
        self._finalize_training(history, training_time, best_model_path)
        
        return history
    
    def _train_epoch(self, dataloader, accumulation_steps):
        """
        训练一个epoch
        """
        self.model.train()
        total_loss = 0
        steps = 0
        
        # 使用tqdm显示进度条
        train_iterator = tqdm(dataloader, desc="Training")
        
        # 梯度累积相关变量
        self.optimizer.zero_grad()
        accumulated_loss = 0
        
        for batch_idx, batch in enumerate(train_iterator):
            loss = self._process_batch(batch)
            loss = loss / accumulation_steps  # 缩放损失
            loss.backward()
            
            accumulated_loss += loss.item()
            
            # 梯度累积
            if (batch_idx + 1) % accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.training_config['gradient_clip_val']
                )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                total_loss += accumulated_loss
                steps += 1
                accumulated_loss = 0
                
                # 更新进度条
                train_iterator.set_postfix({"loss": total_loss / steps})
        
        return total_loss / steps

    def _print_epoch_summary(self, epoch, epochs, train_loss, val_metrics=None):
        """
        打印每个epoch的训练总结
        
        Args:
            epoch: 当前epoch
            epochs: 总epoch数
            train_loss: 训练损失
            val_metrics: 验证指标字典
        """
        summary = f"\nEpoch {epoch+1}/{epochs}"
        
        # 添加安全类型转换
        try:
            summary += f"\n训练损失: {float(train_loss):.4f}"
        except (TypeError, ValueError):
            summary += f"\n训练损失: {train_loss}"
        
        if val_metrics:
            # 安全地添加验证指标
            for metric_name in ['loss', 'accuracy', 'f1', 'precision', 'recall']:
                if metric_name in val_metrics:
                    value = val_metrics[metric_name]
                    # 检查是否为数组
                    if hasattr(value, 'shape') and len(getattr(value, 'shape', [])) > 0:
                        summary += f"\n验证{metric_name}: {value}"
                    else:
                        try:
                            summary += f"\n验证{metric_name}: {float(value):.4f}"
                        except (TypeError, ValueError):
                            summary += f"\n验证{metric_name}: {value}"
            
            # 添加预测分布信息
            if 'predictions' in val_metrics:
                unique, counts = np.unique(val_metrics['predictions'], return_counts=True)
                dist = dict(zip(unique, counts))
                summary += f"\n预测分布: 类别0={dist.get(0, 0)}, 类别1={dist.get(1, 0)}"
        
        logger.info(summary)
        print(summary)

    def _validate(self, dataloader):
        """
        验证模型
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probas = []
        steps = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                loss, batch_preds, batch_labels, batch_probas = self._process_batch(batch, training=False)
                total_loss += loss.item()
                steps += 1
                
                all_preds.extend(batch_preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                if batch_probas is not None:
                    all_probas.extend(batch_probas.cpu().numpy())
        
        # 计算评估指标
        metrics = self._calculate_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / steps
        metrics['predictions'] = all_preds
        if all_probas:
            metrics['probabilities'] = all_probas
        
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred):
        """
        计算评估指标
        """
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, 
            average='binary',
            zero_division=0  # 当没有预测样本时返回0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _log_metrics(self, epoch, train_loss, val_metrics):
        """
        记录指标到TensorBoard
        """
        # 安全添加训练损失
        try:
            self.writer.add_scalar('Loss/train', float(train_loss), epoch)
        except (TypeError, ValueError):
            logger.warning(f"无法记录训练损失 {train_loss} 到TensorBoard，类型不兼容")
            
        for metric, value in val_metrics.items():
            # 只记录数值类型的指标
            if metric in ['loss', 'accuracy', 'precision', 'recall', 'f1'] and not isinstance(value, (list, dict)):
                try:
                    # 确保值是浮点数
                    float_value = float(value)
                    self.writer.add_scalar(f'Metrics/{metric}', float_value, epoch)
                except (TypeError, ValueError):
                    logger.warning(f"无法将指标 {metric} 的值 {value} 转换为浮点数，跳过记录")
    
    def _save_checkpoint(self, epoch, val_f1):
        """
        保存模型检查点
        """
        # 安全格式化 val_f1
        try:
            f1_formatted = f"{float(val_f1):.4f}"
        except (TypeError, ValueError):
            f1_formatted = str(val_f1)
            
        checkpoint_path = os.path.join(
            self.output_dir, 
            'checkpoints', 
            f'model_epoch_{epoch}_f1_{f1_formatted}.pt'
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_config': self.training_config
        }, checkpoint_path)
        
        return checkpoint_path
    
    def _finalize_training(self, history, training_time, best_model_path):
        """
        完成训练后的处理
        """
        # 转换NumPy类型为Python原生类型
        def convert_to_serializable(obj):
            import numpy as np
            if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64,
                np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            return obj

        # 转换历史数据
        serializable_history = convert_to_serializable(history)
        
        # 保存训练历史
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=4)
        
        # 生成训练报告
        self._generate_training_report(history, training_time, best_model_path)
        
        # 绘制训练曲线
        self._plot_training_curves(history)
        
        # 关闭TensorBoard writer
        self.writer.close()
    
    def _generate_training_report(self, history, training_time, best_model_path):
        """
        生成训练报告
        """
        report = {
            'training_time': f"{training_time/60:.2f} minutes",
            'best_model_path': best_model_path,
            'final_metrics': {
                'train_loss': history['train_loss'][-1],
                'val_loss': history['val_loss'][-1] if 'val_loss' in history else None,
                'val_f1': history['val_f1'][-1] if 'val_f1' in history else None
            },
            'training_config': self.training_config
        }
        
        report_path = os.path.join(self.output_dir, 'training_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
    
    def _plot_training_curves(self, history):
        """
        Plot training curves
        """
        plt.figure(figsize=(15, 10))
        
        # Loss curves
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss Curves')
        plt.legend()
        
        # Evaluation metrics
        plt.subplot(2, 2, 2)
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            if f'val_{metric}' in history:
                plt.plot(history[f'val_{metric}'], label=metric.capitalize())
        plt.title('Evaluation Metrics')
        plt.legend()
        
        # Learning rate curve
        plt.subplot(2, 2, 3)
        plt.plot(history['learning_rates'])
        plt.title('Learning Rate')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'training_curves.png'))
        plt.close() 