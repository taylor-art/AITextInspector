import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AITextDetectorEvaluator:
    """
    AI文本检测模型评估器
    """
    def __init__(self, output_dir="./evaluation"):
        """
        初始化评估器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def evaluate_model(self, trainer, test_dataset, batch_size=16):
        """
        评估模型性能
        
        Args:
            trainer: 模型训练器
            test_dataset: 测试数据集
            batch_size: 批量大小
            
        Returns:
            dict: 评估指标
        """
        logger.info("开始评估模型性能")
        metrics = trainer.evaluate(eval_dataset=test_dataset, batch_size=batch_size)
        logger.info(f"评估结果: {metrics}")
        
        return metrics
    
    def plot_confusion_matrix(self, cm, classes=['人类撰写', 'AI生成'], title='混淆矩阵', save_path=None):
        """
        绘制混淆矩阵
        
        Args:
            cm: 混淆矩阵
            classes: 类别名称
            title: 图表标题
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 8))
        sns.set(font_scale=1.2)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title(title)
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        
        plt.savefig(save_path)
        plt.close()
        logger.info(f"混淆矩阵已保存到: {save_path}")
    
    def plot_roc_curve(self, y_true, y_score, save_path=None):
        """
        绘制ROC曲线
        
        Args:
            y_true: 真实标签
            y_score: 预测分数
            save_path: 保存路径
        """
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('接收者操作特征曲线')
        plt.legend(loc="lower right")
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'roc_curve.png')
        
        plt.savefig(save_path)
        plt.close()
        logger.info(f"ROC曲线已保存到: {save_path}")
    
    def plot_training_history(self, history, save_path=None):
        """
        绘制训练历史
        
        Args:
            history: 训练历史
            save_path: 保存路径
        """
        plt.figure(figsize=(12, 4))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='训练损失')
        if 'val_loss' in history and history['val_loss']:
            plt.plot(history['val_loss'], label='验证损失')
        plt.title('训练和验证损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        
        # 绘制指标曲线
        plt.subplot(1, 2, 2)
        if 'val_accuracy' in history and history['val_accuracy']:
            plt.plot(history['val_accuracy'], label='准确率')
        if 'val_f1' in history and history['val_f1']:
            plt.plot(history['val_f1'], label='F1分数')
        plt.title('验证指标')
        plt.xlabel('轮次')
        plt.ylabel('分数')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'training_history.png')
        
        plt.savefig(save_path)
        plt.close()
        logger.info(f"训练历史已保存到: {save_path}")
    
    def print_classification_report(self, y_true, y_pred, target_names=['人类撰写', 'AI生成']):
        """
        打印分类报告
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            target_names: 类别名称
        """
        report = classification_report(y_true, y_pred, target_names=target_names)
        logger.info(f"分类报告:\n{report}")
        
        # 保存到文件
        report_path = os.path.join(self.output_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"分类报告已保存到: {report_path}")
    
    def save_evaluation_results(self, metrics, file_name='evaluation_results.txt'):
        """
        保存评估结果
        
        Args:
            metrics: 评估指标
            file_name: 文件名
        """
        file_path = os.path.join(self.output_dir, file_name)
        
        with open(file_path, 'w') as f:
            f.write("模型评估结果\n")
            f.write("=================\n\n")
            
            for key, value in metrics.items():
                if key != 'confusion_matrix':
                    f.write(f"{key}: {value}\n")
            
            if 'confusion_matrix' in metrics:
                f.write("\n混淆矩阵:\n")
                f.write(str(metrics['confusion_matrix']))
        
        logger.info(f"评估结果已保存到: {file_path}") 