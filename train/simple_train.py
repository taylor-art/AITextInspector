import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import json
import time
from src.utils import get_logger

# 创建日志记录器
logger = get_logger(
    log_dir="./logs",
    experiment_name=f"simple_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    enable_tensorboard=True
)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data():
    """加载数据集"""
    logger.info("开始加载数据集...")
    
    # 加载训练集
    train_df = pd.read_csv('../dataset/AI-and-Human-Generated-Text/train.csv')
    # 合并title和abstract
    train_df['text'] = train_df['title'] + ' ' + train_df['abstract']
    
    # 加载测试集
    test_df = pd.read_csv('../dataset/AI-and-Human-Generated-Text/test.csv')
    test_df['text'] = test_df['title'] + ' ' + test_df['abstract']
    
    # 记录数据集基本统计信息
    data_stats = {
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "train_class_distribution": train_df['label'].value_counts().to_dict(),
        "test_class_distribution": test_df['label'].value_counts().to_dict()
    }
    
    logger.info("数据集统计信息:")
    logger.info(f"训练集样本数: {data_stats['train_samples']}")
    logger.info(f"测试集样本数: {data_stats['test_samples']}")
    logger.info(f"训练集类别分布: {data_stats['train_class_distribution']}")
    logger.info(f"测试集类别分布: {data_stats['test_class_distribution']}")

    # 可视化类别分布
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.countplot(x='label', data=train_df)
    plt.title('Training Set Class Distribution')
    plt.xlabel('Class (0: Human Text, 1: AI-generated Text)')
    plt.ylabel('Sample Count')
    
    plt.subplot(1, 2, 2)
    sns.countplot(x='label', data=test_df)
    plt.title('Test Set Class Distribution')
    plt.xlabel('Class (0: Human Text, 1: AI-generated Text)')
    plt.ylabel('Sample Count')
    
    os.makedirs('output/figures', exist_ok=True)
    plt.tight_layout()
    plt.savefig('output/figures/class_distribution.png', dpi=300)
    plt.close()
    
    logger.info("数据集加载完成，可视化图表已保存")
    
    return (
        train_df['text'].values, train_df['label'].values,
        test_df['text'].values, test_df['label'].values,
        data_stats
    )

def train_model(model, train_loader, test_loader, device, epochs=3, output_dir='output', patience=3):
    """训练模型并记录详细结果"""
    logger.info(f"开始训练模型，总共 {epochs} 个epoch，早停耐心值为 {patience}")
    logger.info(f"输出目录: {output_dir}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    logger.info(f"优化器: AdamW，学习率: 2e-5")
    
    # 创建结果存储目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/figures', exist_ok=True)
    
    # 初始化结果记录
    training_stats = {
        'epochs': [],
        'train_loss': [],
        'train_accuracy': [],
        'test_accuracy': [],
        'test_precision': [],
        'test_recall': [],
        'test_f1': [],
        'test_auc': []
    }
    
    # 记录最佳结果
    best_test_acc = 0
    best_epoch = 0
    best_test_metrics = {}
    
    total_train_time = 0
    early_stopped = False
    manually_stopped = False
    
    try:
        for epoch in range(epochs):
            logger.info(f'\nEpoch {epoch + 1}/{epochs}')
            epoch_start_time = time.time()
            
            # 训练阶段
            model.train()
            train_loss = 0
            train_preds, train_labels = [], []
            
            for batch in tqdm(train_loader, desc='Training'):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                train_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                train_preds.extend(preds)
                train_labels.extend(labels.cpu().numpy())
            
            train_acc = accuracy_score(train_labels, train_preds)
            avg_train_loss = train_loss/len(train_loader)
            
            logger.info(f'Training Loss: {avg_train_loss:.4f}')
            logger.info(f'Training Accuracy: {train_acc:.4f}')
            
            # 评估阶段
            model.eval()
            test_preds, test_labels = [], []
            test_probs = []
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc='Testing'):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    probs = torch.softmax(outputs.logits, dim=1)
                    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                    
                    test_preds.extend(preds)
                    test_labels.extend(labels.cpu().numpy())
                    test_probs.extend(probs[:, 1].cpu().numpy())  # 取类别1的概率
            
            # 计算各种性能指标
            test_acc = accuracy_score(test_labels, test_preds)
            test_report = classification_report(test_labels, test_preds, output_dict=True)
            test_auc = roc_auc_score(test_labels, test_probs)
            
            logger.info(f'Test Accuracy: {test_acc:.4f}')
            logger.info(f'Test AUC: {test_auc:.4f}')
            logger.info('\nClassification Report:')
            logger.info(classification_report(test_labels, test_preds))
            
            # 计算当前epoch的训练时间
            epoch_time = time.time() - epoch_start_time
            total_train_time += epoch_time
            logger.info(f'Epoch {epoch+1} training time: {epoch_time:.2f} seconds')
            
            # 记录当前epoch的结果
            epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'test_precision': test_report['weighted avg']['precision'],
                'test_recall': test_report['weighted avg']['recall'],
                'test_f1': test_report['weighted avg']['f1-score'],
                'test_auc': test_auc,
                'epoch_time': epoch_time
            }
            
            # 更新训练统计信息
            for key in training_stats:
                if key != 'epochs':
                    training_stats[key].append(epoch_stats.get(key, None))
            training_stats['epochs'].append(epoch + 1)
            
            # 保存最佳模型
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch + 1
                best_test_metrics = {
                    'accuracy': test_acc,
                    'precision': test_report['weighted avg']['precision'],
                    'recall': test_report['weighted avg']['recall'],
                    'f1': test_report['weighted avg']['f1-score'],
                    'auc': test_auc,
                    'confusion_matrix': confusion_matrix(test_labels, test_preds).tolist(),
                    'classification_report': test_report
                }
                
                # 保存最佳模型
                model.save_pretrained(f'{output_dir}/best_model')
                logger.info(f"Saved best model with accuracy: {test_acc:.4f}")
            
            # 绘制每个epoch的混淆矩阵
            plot_confusion_matrix(test_labels, test_preds, epoch, output_dir=f'{output_dir}/figures')
            
            # 每个epoch后绘制ROC曲线
            plot_roc_curve(test_labels, test_probs, epoch, output_dir=f'{output_dir}/figures')
            
            # 检查早停
            if epoch >= patience:
                no_improvement = True
                # 检查最近几个epoch是否有性能提升
                for i in range(1, patience + 1):
                    if epoch - i + 1 == best_epoch:
                        no_improvement = False
                        break
                
                if no_improvement:
                    logger.info(f"Early stopping at epoch {epoch + 1} - No improvement for {patience} epochs")
                    early_stopped = True
                    break
                    
            # 每个epoch结束提示用户可以按Ctrl+C停止训练
            logger.info("Training in progress... Press Ctrl+C to stop")
    
    except KeyboardInterrupt:
        logger.info("\nTraining manually stopped by user")
        manually_stopped = True
    
    # 训练结束后的总结
    logger.info("\n========== Training Completed ==========")
    if early_stopped:
        logger.info(f"Early stopped at epoch {len(training_stats['epochs'])}/{epochs}")
    elif manually_stopped:
        logger.info(f"Manually stopped at epoch {len(training_stats['epochs'])}/{epochs}")
    else:
        logger.info(f"Completed all {epochs} epochs")
        
    logger.info(f"Total Training Time: {total_train_time:.2f} seconds")
    logger.info(f"Best Model (Epoch {best_epoch}):")
    logger.info(f"  Accuracy: {best_test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {best_test_metrics['precision']:.4f}")
    logger.info(f"  Recall: {best_test_metrics['recall']:.4f}")
    logger.info(f"  F1 Score: {best_test_metrics['f1']:.4f}")
    logger.info(f"  AUC: {best_test_metrics['auc']:.4f}")
    
    # 绘制训练过程中的指标变化
    plot_training_metrics(training_stats, output_dir=f'{output_dir}/figures')
    
    # 保存训练统计信息
    training_summary = {
        'total_epochs_trained': len(training_stats['epochs']),
        'total_epochs_planned': epochs,
        'early_stopped': early_stopped,
        'manually_stopped': manually_stopped,
        'best_epoch': best_epoch,
        'training_time': total_train_time,
        'stats': training_stats
    }
    
    with open(f'{output_dir}/training_stats.json', 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    # 保存最佳测试指标
    with open(f'{output_dir}/best_test_metrics.json', 'w') as f:
        json.dump(best_test_metrics, f, indent=2)
    
    return model, training_summary, best_test_metrics

def plot_confusion_matrix(y_true, y_pred, epoch, output_dir='output/figures'):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Human Text', 'AI-generated Text'],
                yticklabels=['Human Text', 'AI-generated Text'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Epoch {epoch+1} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix_epoch_{epoch+1}.png', dpi=300)
    plt.close()

def plot_roc_curve(y_true, y_prob, epoch, output_dir='output/figures'):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'Epoch {epoch+1} ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curve_epoch_{epoch+1}.png', dpi=300)
    plt.close()
    
    # 同时绘制精确率-召回率曲线
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AP = {avg_precision:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Epoch {epoch+1} Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pr_curve_epoch_{epoch+1}.png', dpi=300)
    plt.close()

def plot_training_metrics(training_stats, output_dir='output/figures'):
    """绘制训练过程中的各项指标变化"""
    epochs = training_stats['epochs']
    
    # 绘制损失和准确率
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_stats['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_stats['train_accuracy'], label='Training Accuracy')
    plt.plot(epochs, training_stats['test_accuracy'], label='Test Accuracy')
    plt.title('Training/Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/loss_accuracy.png', dpi=300)
    plt.close()
    
    # 绘制其他测试指标
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_stats['test_precision'], label='Precision')
    plt.plot(epochs, training_stats['test_recall'], label='Recall')
    plt.plot(epochs, training_stats['test_f1'], label='F1 Score')
    plt.title('Test Set Performance Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_stats['test_auc'])
    plt.title('Test Set AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/test_metrics.png', dpi=300)
    plt.close()
    
    # 添加学习曲线图 - 训练集与测试集对比
    plt.figure(figsize=(15, 10))
    
    # 1. 准确率学习曲线
    plt.subplot(2, 2, 1)
    plt.plot(epochs, training_stats['train_accuracy'], marker='o', label='Training Accuracy')
    plt.plot(epochs, training_stats['test_accuracy'], marker='s', label='Test Accuracy')
    plt.title('Accuracy Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 2. 损失与AUC对比
    plt.subplot(2, 2, 2)
    fig, ax1 = plt.subplots(figsize=(15, 10))
    plt.subplot(2, 2, 2)
    plt.plot(epochs, training_stats['train_loss'], 'b-', marker='o', label='Training Loss')
    plt.ylabel('Loss', color='b')
    plt.tick_params(axis='y', labelcolor='b')
    plt.legend(loc='upper left')
    plt.grid(True)
    
    ax2 = plt.twinx()
    ax2.plot(epochs, training_stats['test_auc'], 'r-', marker='s', label='Test AUC')
    ax2.set_ylabel('AUC', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc='upper right')
    plt.title('Loss and AUC')
    plt.xlabel('Epochs')
    
    # 3. 精确率和召回率
    plt.subplot(2, 2, 3)
    plt.plot(epochs, training_stats['test_precision'], marker='o', label='Precision')
    plt.plot(epochs, training_stats['test_recall'], marker='s', label='Recall')
    plt.title('Precision and Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    # 4. F1分数
    plt.subplot(2, 2, 4)
    plt.plot(epochs, training_stats['test_f1'], marker='o', color='purple')
    plt.title('F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/learning_curves.png', dpi=300)
    plt.close()

def error_analysis(true_labels, pred_labels, texts):
    """对错误分类的样本进行分析"""
    logger.info("执行错误分析...")
    
    misclassified = []
    for idx, (true, pred, text) in enumerate(zip(true_labels, pred_labels, texts)):
        if true != pred:
            misclassified.append({
                'id': idx,
                'true_label': true,
                'pred_label': pred,
                'text': text[:100] + '...' if len(text) > 100 else text
            })
    
    error_stats = {
        'total_samples': len(true_labels),
        'misclassified_count': len(misclassified),
        'error_rate': len(misclassified) / len(true_labels) if len(true_labels) > 0 else 0,
        'false_positives': sum(1 for m in misclassified if m['true_label'] == 0 and m['pred_label'] == 1),
        'false_negatives': sum(1 for m in misclassified if m['true_label'] == 1 and m['pred_label'] == 0)
    }
    
    # 保存错误分类的样本
    misclassified_df = pd.DataFrame(misclassified)
    if not misclassified_df.empty:
        os.makedirs('output/error_analysis', exist_ok=True)
        misclassified_df.to_csv('output/error_analysis/misclassified_samples.csv', index=False)
        logger.info(f"已将{len(misclassified)}个错误分类样本保存到 output/error_analysis/misclassified_samples.csv")
    
    # 创建混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Human', 'AI'],
                yticklabels=['Human', 'AI'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    os.makedirs('output/figures', exist_ok=True)
    plt.savefig('output/figures/confusion_matrix.png')
    logger.info("混淆矩阵已保存到 output/figures/confusion_matrix.png")
    
    # 计算并显示每个类别的指标
    report = classification_report(true_labels, pred_labels, output_dict=True)
    logger.info("\n每个类别的性能指标:")
    logger.info(f"Human类别 - Precision: {report['0']['precision']:.4f}, Recall: {report['0']['recall']:.4f}, F1: {report['0']['f1-score']:.4f}")
    logger.info(f"AI类别 - Precision: {report['1']['precision']:.4f}, Recall: {report['1']['recall']:.4f}, F1: {report['1']['f1-score']:.4f}")
    
    return misclassified, error_stats

def analyze_misclassifications(model, tokenizer, X_test, y_test, device='cpu'):
    """深入分析错误分类的样本"""
    logger.info("开始分析错误分类样本...")
    
    model.eval()
    all_preds = []
    all_probs = []
    
    # 使用较小的批次以防止内存溢出
    batch_size = 16
    dataset = TextDataset(X_test, y_test, tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Analyzing'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # 计算性能指标
    accuracy = accuracy_score(y_test, all_preds)
    auc = roc_auc_score(y_test, all_probs)
    
    logger.info(f"全量测试集性能: 准确率: {accuracy:.4f}, AUC: {auc:.4f}")
    
    # 错误分析
    misclassified, error_stats = error_analysis(y_test, all_preds, X_test)
    
    # 找出置信度最高的错误分类样本
    if misclassified:
        # 为错误分类的样本添加概率信息
        for i, m in enumerate(misclassified):
            sample_idx = m['id']
            m['confidence'] = all_probs[sample_idx]
        
        # 按置信度排序
        misclassified.sort(key=lambda x: abs(x['confidence'] - 0.5), reverse=True)
        
        # 显示最有信心的5个错误预测
        logger.info("\n置信度最高的误分类样本:")
        confident_errors = misclassified[:5]
        for i, sample in enumerate(confident_errors):
            true_label = "Human" if sample['true_label'] == 0 else "AI"
            pred_label = "Human" if sample['pred_label'] == 0 else "AI"
            logger.info(f"样本 {i+1}: 真实标签: {true_label}, 预测标签: {pred_label}, 置信度: {sample['confidence']:.4f}")
            logger.info(f"文本片段: {sample['text'][:200]}...")
            logger.info("-"*50)
    
    return misclassified, error_stats

def plot_final_evaluation(best_metrics, test_loader, model, device, output_dir='output/figures'):
    """生成最终模型评估的详细可视化图表"""
    logger.info("生成最终评估可视化图表...")
    
    # 获取测试集结果
    model.eval()
    all_preds = []
    all_probs = []
    all_true = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_true.extend(labels.cpu().numpy())
    
    # 1. 混淆矩阵热力图 - 更详细版本
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_true, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 绘制原始混淆矩阵
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Human Text', 'AI-generated Text'],
                yticklabels=['Human Text', 'AI-generated Text'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Counts)')
    
    # 绘制归一化混淆矩阵
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['Human Text', 'AI-generated Text'],
                yticklabels=['Human Text', 'AI-generated Text'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Normalized)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/final_confusion_matrix.png', dpi=300)
    plt.close()
    logger.info(f"混淆矩阵已保存至 {output_dir}/final_confusion_matrix.png")
    
    # 2. ROC曲线与PR曲线对比
    plt.figure(figsize=(16, 6))
    
    # ROC曲线
    plt.subplot(1, 2, 1)
    fpr, tpr, thresholds_roc = roc_curve(all_true, all_probs)
    auc = roc_auc_score(all_true, all_probs)
    
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    
    # 找出最佳阈值点 (距离左上角最近的点)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds_roc[optimal_idx]
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', 
             label=f'Best Threshold = {optimal_threshold:.2f}')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    
    # PR曲线
    plt.subplot(1, 2, 2)
    precision, recall, thresholds_pr = precision_recall_curve(all_true, all_probs)
    avg_precision = average_precision_score(all_true, all_probs)
    
    plt.plot(recall, precision, label=f'AP = {avg_precision:.4f}')
    
    # 尝试找出PR曲线的最佳阈值点 (F1最大的点)
    f1_scores = np.nan_to_num(2 * precision * recall / (precision + recall), 0)
    best_f1_idx = np.argmax(f1_scores)
    
    if len(thresholds_pr) > best_f1_idx:
        best_threshold_pr = thresholds_pr[best_f1_idx]
        plt.plot(recall[best_f1_idx], precision[best_f1_idx], 'ro', 
                label=f'Best F1 Threshold = {best_threshold_pr:.2f}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/final_roc_pr_curves.png', dpi=300)
    plt.close()
    logger.info(f"ROC和PR曲线已保存至 {output_dir}/final_roc_pr_curves.png")
    
    # 3. 阈值分析图 - 根据不同阈值展示准确率、精确率、召回率和F1值
    plt.figure(figsize=(15, 10))
    
    # 计算不同阈值下的性能指标
    thresholds = np.linspace(0, 1, 100)
    accuracy = []
    precision_list = []
    recall_list = []
    f1_list = []
    
    for threshold in thresholds:
        preds_at_threshold = (np.array(all_probs) >= threshold).astype(int)
        accuracy.append(accuracy_score(all_true, preds_at_threshold))
        
        # 处理可能的警告 (当某个类别在预测中不存在时)
        try:
            precision_list.append(precision_score(all_true, preds_at_threshold, zero_division=0))
        except:
            precision_list.append(0)
            
        try:
            recall_list.append(recall_score(all_true, preds_at_threshold, zero_division=0))
        except:
            recall_list.append(0)
            
        try:
            f1_list.append(f1_score(all_true, preds_at_threshold, zero_division=0))
        except:
            f1_list.append(0)
    
    # 绘制阈值分析图
    plt.subplot(2, 2, 1)
    plt.plot(thresholds, accuracy, marker='.')
    plt.title('Accuracy vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(thresholds, precision_list, marker='.')
    plt.title('Precision vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(thresholds, recall_list, marker='.')
    plt.title('Recall vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Recall')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(thresholds, f1_list, marker='.')
    plt.title('F1 Score vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/threshold_analysis.png', dpi=300)
    plt.close()
    logger.info(f"阈值分析图已保存至 {output_dir}/threshold_analysis.png")
    
    # 4. 组合性能图 - 在一张图上显示所有指标
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, accuracy, label='Accuracy')
    plt.plot(thresholds, precision_list, label='Precision')
    plt.plot(thresholds, recall_list, label='Recall')
    plt.plot(thresholds, f1_list, label='F1 Score')
    
    # 找出F1分数最大的阈值
    best_f1_threshold = thresholds[np.argmax(f1_list)]
    plt.axvline(x=best_f1_threshold, color='r', linestyle='--', 
               label=f'Best F1 Threshold: {best_f1_threshold:.2f}')
    
    plt.title('Model Performance vs Classification Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/combined_metrics_vs_threshold.png', dpi=300)
    plt.close()
    logger.info(f"组合性能图已保存至 {output_dir}/combined_metrics_vs_threshold.png")
    
    # 5. 预测概率分布图
    plt.figure(figsize=(12, 5))
    
    # 按真实标签分组绘制概率分布
    human_probs = [prob for prob, label in zip(all_probs, all_true) if label == 0]
    ai_probs = [prob for prob, label in zip(all_probs, all_true) if label == 1]
    
    plt.subplot(1, 2, 1)
    plt.hist(human_probs, bins=50, alpha=0.5, label='Human Texts')
    plt.hist(ai_probs, bins=50, alpha=0.5, label='AI-generated Texts')
    plt.xlabel('Predicted Probability of being AI-generated')
    plt.ylabel('Count')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    plt.grid(True)
    
    # 箱形图 - 修复matplotlib警告
    plt.subplot(1, 2, 2)
    data = [human_probs, ai_probs]
    try:
        # 尝试使用新参数名 (matplotlib 3.9+)
        plt.boxplot(data, tick_labels=['Human Texts', 'AI-generated Texts'])
    except TypeError:
        # 如果失败，回退到旧参数名
        plt.boxplot(data, labels=['Human Texts', 'AI-generated Texts'])
    
    plt.ylabel('Predicted Probability of being AI-generated')
    plt.title('Prediction Probability Boxplot')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/prediction_probability_distribution.png', dpi=300)
    plt.close()
    logger.info(f"预测概率分布图已保存至 {output_dir}/prediction_probability_distribution.png")
    
    # 记录最终的评估结果
    best_threshold_info = {
        'roc_best_threshold': float(optimal_threshold),
        'f1_best_threshold': float(best_f1_threshold),
    }
    
    with open(f'{output_dir[:-8]}/threshold_analysis.json', 'w') as f:
        json.dump(best_threshold_info, f, indent=2)
    
    logger.info(f"最佳阈值分析: ROC最佳阈值 = {optimal_threshold:.4f}, F1最佳阈值 = {best_f1_threshold:.4f}")
    logger.info("所有评估图表生成完成")
    
    return best_threshold_info

def main():
    # 设置随机种子以确保可重复性
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'output/run_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/figures', exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # 使用更轻量级的模型
    try:
        logger.info("Using lightweight model: distilbert-base-uncased")
        from transformers import DistilBertForSequenceClassification
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = DistilBertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2,
            problem_type="single_label_classification"
        ).to(device)
        logger.info("Successfully loaded lightweight model!")
    except Exception as e:
        logger.info(f"Error loading lightweight model: {e}")
        logger.info("Trying alternative model: bert-base-uncased")
        
        from transformers import BertForSequenceClassification
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2,
            problem_type="single_label_classification"
        ).to(device)
    
    # 获取和打印模型参数数量
    model_size = sum(p.numel() for p in model.parameters())
    logger.info(f'Model parameters: {model_size:,}')
    
    # 加载数据
    logger.info('Loading dataset...')
    X_train, y_train, X_test, y_test, data_stats = load_data()
    
    logger.info(f"Training with full dataset: {len(X_train)} training samples and {len(X_test)} test samples")
    
    # 创建数据集
    train_dataset = TextDataset(X_train, y_train, tokenizer)
    test_dataset = TextDataset(X_test, y_test, tokenizer)
    
    # 创建数据加载器 - 使用更合适的批次大小
    batch_size = 32  # 增加批次大小提高训练效率
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 训练模型 - 增加epoch数量并添加早停
    logger.info('Starting model training...')
    model, training_summary, best_test_metrics = train_model(
        model, 
        train_loader, 
        test_loader, 
        device, 
        epochs=15,  # 增加训练轮数为15
        output_dir=output_dir,
        patience=5  # 增加早停耐心值为5
    )
    
    # 生成最终评估图表
    logger.info('Generating detailed evaluation charts...')
    threshold_info = plot_final_evaluation(
        best_test_metrics,
        test_loader,
        model,
        device,
        output_dir=f'{output_dir}/figures'
    )
    
    # 分析错误分类的样本
    logger.info('Analyzing misclassifications...')
    misclassified, error_stats = analyze_misclassifications(
        model, 
        tokenizer, 
        X_test, 
        y_test, 
        device
    )
    
    # 保存完整的实验配置和结果摘要
    experiment_summary = {
        'timestamp': timestamp,
        'model_name': model_name,
        'model_parameters': model_size,
        'data_stats': data_stats,
        'training_config': {
            'batch_size': batch_size,
            'epochs': 15,
            'optimizer': 'AdamW',
            'learning_rate': 2e-5,
            'seed': seed,
            'device': str(device),
            'max_length': 128,
            'quick_test': False,
            'data_fraction': 1.0,
            'patience': 5
        },
        'training_summary': training_summary,
        'best_results': best_test_metrics,
        'threshold_analysis': threshold_info,
        'error_analysis': error_stats
    }
    
    with open(f'{output_dir}/experiment_summary.json', 'w') as f:
        json.dump(experiment_summary, f, indent=2)
    
    # 保存模型和分词器
    logger.info(f'Saving model to {output_dir}/final_model')
    model.save_pretrained(f'{output_dir}/final_model')
    tokenizer.save_pretrained(f'{output_dir}/final_model')
    
    logger.info(f'All results saved to {output_dir}')
    logger.info('Training completed!')

if __name__ == '__main__':
    main() 