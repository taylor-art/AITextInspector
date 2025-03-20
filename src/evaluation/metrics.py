"""
评估指标模块
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import logging
import os

logger = logging.getLogger(__name__)

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    计算模型性能指标
    
    Args:
        y_true (array): 真实标签
        y_pred (array): 预测标签
        y_pred_proba (array, optional): 预测概率，用于计算AUC
        
    Returns:
        dict: 包含各项指标的字典
    """
    metrics = {}
    
    # 计算基本指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='binary')
    metrics['recall'] = recall_score(y_true, y_pred, average='binary')
    metrics['f1'] = f1_score(y_true, y_pred, average='binary')
    
    # 如果提供了概率，计算AUC
    if y_pred_proba is not None:
        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
    
    # 计算混淆矩阵
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    """
    绘制混淆矩阵
    
    Args:
        y_true (array): 真实标签
        y_pred (array): 预测标签
        title (str): 图标题
        save_path (str, optional): 保存路径
        
    Returns:
        plt.Figure: matplotlib图对象
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title, fontsize=16)
    plt.xlabel('预测标签', fontsize=14)
    plt.ylabel('真实标签', fontsize=14)
    
    # 添加类别标签
    plt.xticks([0.5, 1.5], ['人类撰写 (0)', 'AI生成 (1)'], fontsize=12)
    plt.yticks([0.5, 1.5], ['人类撰写 (0)', 'AI生成 (1)'], fontsize=12)
    
    # 添加注释
    plt.text(0.5, 2.5, f'准确率: {(cm[0, 0] + cm[1, 1]) / cm.sum():.4f}', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"混淆矩阵已保存到: {save_path}")
    
    return plt.gcf()

def plot_roc_curve(y_true, y_pred_proba, title="ROC曲线", save_path=None):
    """
    绘制ROC曲线
    
    Args:
        y_true (array): 真实标签
        y_pred_proba (array): 预测概率
        title (str): 图标题
        save_path (str, optional): 保存路径
        
    Returns:
        plt.Figure: matplotlib图对象
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率', fontsize=14)
    plt.ylabel('真阳性率', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC曲线已保存到: {save_path}")
    
    return plt.gcf()

def plot_precision_recall_curve(y_true, y_pred_proba, title="精确率-召回率曲线", save_path=None):
    """
    绘制精确率-召回率曲线
    
    Args:
        y_true (array): 真实标签
        y_pred_proba (array): 预测概率
        title (str): 图标题
        save_path (str, optional): 保存路径
        
    Returns:
        plt.Figure: matplotlib图对象
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='darkorange', lw=2, 
             label=f'精确率-召回率曲线 (AP = {avg_precision:.4f})')
    plt.xlabel('召回率', fontsize=14)
    plt.ylabel('精确率', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc="best", fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"精确率-召回率曲线已保存到: {save_path}")
    
    return plt.gcf()

def plot_training_history(history, metrics=None, title="训练历史", save_path=None):
    """
    绘制训练历史
    
    Args:
        history (dict): 训练历史字典
        metrics (list, optional): 要绘制的指标列表
        title (str): 图标题
        save_path (str, optional): 保存路径
        
    Returns:
        plt.Figure: matplotlib图对象
    """
    if metrics is None:
        metrics = ['loss', 'accuracy']
    
    plt.figure(figsize=(12, 8))
    
    for i, metric in enumerate(metrics):
        plt.subplot(1, len(metrics), i+1)
        plt.plot(history[metric], label=f'训练{metric}')
        plt.plot(history[f'val_{metric}'], label=f'验证{metric}')
        plt.title(f'{metric.capitalize()}')
        plt.xlabel('轮次')
        plt.ylabel(metric.capitalize())
        plt.legend()
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"训练历史图已保存到: {save_path}")
    
    return plt.gcf()

def plot_feature_importance(model, feature_names=None, top_n=20, title="特征重要性", save_path=None):
    """
    绘制特征重要性图
    
    Args:
        model: 模型对象，必须有feature_importances_属性
        feature_names (list, optional): 特征名称列表
        top_n (int): 显示的顶部特征数量
        title (str): 图标题
        save_path (str, optional): 保存路径
        
    Returns:
        plt.Figure: matplotlib图对象
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("模型没有feature_importances_属性，无法绘制特征重要性图")
        return None
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importances))]
    
    plt.figure(figsize=(12, 8))
    plt.title(title, fontsize=16)
    plt.bar(range(top_n), importances[indices], align='center')
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, top_n])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"特征重要性图已保存到: {save_path}")
    
    return plt.gcf()

def evaluate_model(model, X_test, y_test, output_dir=None):
    """
    评估模型并生成完整的评估报告
    
    Args:
        model: 模型对象
        X_test: 测试特征
        y_test: 测试标签
        output_dir (str, optional): 输出目录
        
    Returns:
        dict: 评估指标
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 获取预测概率（如果模型支持）
    y_pred_proba = None
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 计算指标
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    
    # 打印分类报告
    logger.info("\n" + classification_report(y_test, y_pred))
    
    # 可视化
    if output_dir:
        # 绘制混淆矩阵
        plot_confusion_matrix(
            y_test, y_pred, 
            title="混淆矩阵", 
            save_path=os.path.join(output_dir, "confusion_matrix.png")
        )
        
        # 绘制ROC曲线（如果有概率）
        if y_pred_proba is not None:
            plot_roc_curve(
                y_test, y_pred_proba, 
                title="ROC曲线", 
                save_path=os.path.join(output_dir, "roc_curve.png")
            )
            
            # 绘制精确率-召回率曲线
            plot_precision_recall_curve(
                y_test, y_pred_proba, 
                title="精确率-召回率曲线", 
                save_path=os.path.join(output_dir, "pr_curve.png")
            )
        
        # 绘制特征重要性（如果模型支持）
        if hasattr(model, 'feature_importances_'):
            plot_feature_importance(
                model, 
                title="特征重要性", 
                save_path=os.path.join(output_dir, "feature_importance.png")
            )
    
    return metrics 