import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import json
import time

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):  # 使用更小的序列长度
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

def load_minimal_data():
    """加载极少量数据用于测试"""
    # 加载训练集，仅读取前20行
    train_df = pd.read_csv('dataset/AI-and-Human-Generated-Text/train.csv', nrows=20)
    train_df['text'] = train_df['title'] + ' ' + train_df['abstract']
    
    # 加载测试集，仅读取前10行
    test_df = pd.read_csv('dataset/AI-and-Human-Generated-Text/test.csv', nrows=10)
    test_df['text'] = test_df['title'] + ' ' + test_df['abstract']
    
    print("极简数据集:")
    print(f"训练样本数: {len(train_df)}")
    print(f"测试样本数: {len(test_df)}")
    
    return (
        train_df['text'].values, train_df['label'].values,
        test_df['text'].values, test_df['label'].values
    )

def train_minimal_model(model, train_loader, test_loader, device):
    """极简训练循环，只训练一个epoch"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    print('开始训练...')
    # 训练阶段
    model.train()
    for batch in tqdm(train_loader, desc='训练中'):
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
        loss.backward()
        optimizer.step()
    
    # 评估阶段
    print('开始评估...')
    model.eval()
    test_preds, test_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='测试中'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(labels.cpu().numpy())
    
    # 计算准确率
    accuracy = accuracy_score(test_labels, test_preds)
    print(f"测试准确率: {accuracy:.4f}")
    
    return model

def main():
    print("="*50)
    print("运行极简测试版本 - 验证流程是否正常")
    print("="*50)
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 使用极轻量级模型
    try:
        print("加载极轻量级模型...")
        from transformers import DistilBertForSequenceClassification
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = DistilBertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2
        ).to(device)
    except Exception as e:
        print(f"加载失败: {e}")
        print("尝试使用备选模型...")
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        ).to(device)
    
    # 获取模型参数数量
    model_size = sum(p.numel() for p in model.parameters())
    print(f'模型参数数量: {model_size:,}')
    
    # 加载极少量数据
    print('加载极少量数据...')
    X_train, y_train, X_test, y_test = load_minimal_data()
    
    # 创建数据集
    print('创建数据集和数据加载器...')
    train_dataset = TextDataset(X_train, y_train, tokenizer)
    test_dataset = TextDataset(X_test, y_test, tokenizer)
    
    # 使用极小批次
    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 训练模型
    start_time = time.time()
    model = train_minimal_model(model, train_loader, test_loader, device)
    training_time = time.time() - start_time
    
    print(f"训练完成! 耗时: {training_time:.2f} 秒")
    print("极简测试成功运行，流程验证通过！")
    print("注意: 此测试仅用于验证流程，结果无统计意义")

if __name__ == '__main__':
    main() 