import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import logging
import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    用于加载和预处理AI文本检测任务的数据集
    """
    def __init__(self, data_dir, max_samples=None):
        """
        初始化DataLoader
        
        Args:
            data_dir (str): 数据集所在目录
            max_samples (int, optional): 最大加载样本数量，用于测试或调试
        """
        self.data_dir = data_dir
        self.max_samples = max_samples
        self.data_files = {
            'combined': os.path.join(data_dir, 'AI-and-Human-Generated-Text', 'combined_ai_gen_dataset.csv'),
            'test': os.path.join(data_dir, 'AI-and-Human-Generated-Text', 'test.csv'),
            'train': os.path.join(data_dir, 'AI-and-Human-Generated-Text', 'train.csv')
        }
        
    def load_file(self, file_type='train'):
        """
        加载指定类型的数据文件
        
        Args:
            file_type (str): 文件类型，可选值: 'combined', 'test', 'train'
            
        Returns:
            pd.DataFrame: 加载的数据
        """
        if file_type not in self.data_files:
            raise ValueError(f"未知的文件类型: {file_type}，可选值: {list(self.data_files.keys())}")
            
        file_path = self.data_files[file_type]
        logger.info(f"加载数据集：{file_path}")
        
        try:
            # 使用chunksize参数分块读取大文件以避免内存问题
            if self.max_samples is not None:
                df = pd.read_csv(file_path, nrows=self.max_samples)
            else:
                # 对于大文件，使用分块读取并合并
                chunks = []
                for chunk in tqdm.tqdm(pd.read_csv(file_path, chunksize=10000), 
                                      desc=f"加载{file_type}数据"):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            
            # 清理数据集，删除无用的列（如Unnamed列）
            if file_type == 'combined':
                # 仅保留abstract和label列
                columns_to_keep = ['abstract', 'label']
                columns_to_keep = [col for col in columns_to_keep if col in df.columns]
                df = df[columns_to_keep]
            
            logger.info(f"成功加载数据，样本数量：{len(df)}")
            return df
        except Exception as e:
            logger.error(f"加载数据时出错：{e}")
            raise
            
    def load_data(self, data_type='train'):
        """
        加载训练数据，支持不同数据类型
        
        Args:
            data_type (str): 数据类型，可选值: 'combined', 'test', 'train'
            
        Returns:
            pd.DataFrame: 加载的数据
        """
        return self.load_file(data_type)
            
    def get_column_mappings(self, df):
        """
        获取不同数据集的列名映射
        
        Args:
            df (pd.DataFrame): 数据集
            
        Returns:
            tuple: (text_column, label_column)
        """
        # 对于新数据集，text列一直是abstract
        text_column = 'abstract'
        
        # 标签列
        label_column = 'label'
            
        logger.info(f"使用列 '{text_column}' 作为文本列，'{label_column}' 作为标签列")
        return text_column, label_column
        
    def prepare_data(self, data_type='train', test_size=0.2, random_state=42, balanced=True):
        """
        准备训练和测试数据集
        
        Args:
            data_type (str): 数据类型，可选值: 'combined', 'test', 'train'
            test_size (float): 测试集比例
            random_state (int): 随机种子
            balanced (bool): 是否平衡正负样本数量
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # 如果指定了test.csv，则使用train.csv作为训练集，test.csv作为测试集
        if data_type == 'test_split':
            train_df = self.load_data('train')
            test_df = self.load_data('test')
            
            # 获取列名映射
            train_text_column, train_label_column = self.get_column_mappings(train_df)
            test_text_column, test_label_column = self.get_column_mappings(test_df)
            
            X_train = train_df[train_text_column].values
            y_train = train_df[train_label_column].values
            X_test = test_df[test_text_column].values
            y_test = test_df[test_label_column].values
            
            logger.info(f"使用预定义拆分：训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
        else:
            # 使用单个数据集并拆分
            df = self.load_data(data_type)
            
            # 获取列名映射
            text_column, label_column = self.get_column_mappings(df)
                
            X = df[text_column].values
            y = df[label_column].values
            
            logger.info(f"数据集特征数量: {len(X)}, 标签数量: {len(y)}")
            logger.info(f"标签分布: 0 (人类撰写) = {sum(y==0)}, 1 (AI生成) = {sum(y==1)}")
            
            # 平衡数据集（可选）
            if balanced:
                # 获取少数类的样本数
                n_min_class = min(sum(y==0), sum(y==1))
                
                # 分别从正样本和负样本中抽取相同数量的样本
                neg_indices = np.where(y==0)[0]
                pos_indices = np.where(y==1)[0]
                
                if len(neg_indices) > n_min_class:
                    neg_indices = np.random.choice(neg_indices, n_min_class, replace=False)
                
                if len(pos_indices) > n_min_class:
                    pos_indices = np.random.choice(pos_indices, n_min_class, replace=False)
                
                # 合并索引并随机打乱
                selected_indices = np.concatenate([neg_indices, pos_indices])
                np.random.shuffle(selected_indices)
                
                X = X[selected_indices]
                y = y[selected_indices]
                
                logger.info(f"平衡后数据集: 特征数量 {len(X)}, 标签分布: 0 = {sum(y==0)}, 1 = {sum(y==1)}")
            
            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
            logger.info(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
            logger.info(f"训练集标签分布: 0 = {sum(y_train==0)}, 1 = {sum(y_train==1)}")
            logger.info(f"测试集标签分布: 0 = {sum(y_test==0)}, 1 = {sum(y_test==1)}")
        
        return X_train, X_test, y_train, y_test
    
    def get_sample_data(self, n_samples=10, data_type='train'):
        """
        获取样本数据用于测试或调试
        
        Args:
            n_samples (int): 样本数量
            data_type (str): 数据类型，可选值: 'combined', 'test', 'train'
            
        Returns:
            pd.DataFrame: 样本数据
        """
        df = self.load_data(data_type)
        return df.sample(n=min(n_samples, len(df)), random_state=42)
    
    def get_dataset_stats(self):
        """
        获取所有数据集的统计信息
        
        Returns:
            dict: 包含各数据集统计信息的字典
        """
        stats = {}
        
        for data_type in self.data_files.keys():
            try:
                df = self.load_data(data_type)
                text_column, label_column = self.get_column_mappings(df)
                
                human_count = sum(df[label_column] == 0)
                ai_count = sum(df[label_column] == 1)
                total = len(df)
                
                # 计算平均文本长度
                avg_length = df[text_column].apply(len).mean()
                
                stats[data_type] = {
                    "总样本数": total,
                    "人类撰写样本数": human_count,
                    "AI生成样本数": ai_count,
                    "人类样本比例": human_count / total if total > 0 else 0,
                    "AI样本比例": ai_count / total if total > 0 else 0,
                    "平均文本长度": avg_length
                }
                
            except Exception as e:
                logger.error(f"获取{data_type}数据集统计信息时出错: {e}")
                stats[data_type] = {"错误": str(e)}
                
        return stats 