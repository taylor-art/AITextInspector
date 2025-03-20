import re
import string
import nltk
import logging
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from transformers import AutoTokenizer
import pandas as pd
import multiprocessing
from functools import partial
from joblib import Parallel, delayed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 下载必要的NLTK资源（如果尚未下载）
def download_nltk_data():
    """下载NLTK必要的数据包"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        logger.info("正在下载NLTK数据包...")
        try:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            logger.info("NLTK数据包下载完成")
        except Exception as e:
            logger.error(f"下载NLTK数据包时出错: {e}")
            logger.error("请手动运行 python -m nltk.downloader punkt stopwords wordnet")
            raise

# 尝试下载NLTK数据
download_nltk_data()

class TextPreprocessor:
    """
    文本预处理工具类，提供多种文本清洗和特征提取方法
    """
    def __init__(self, 
                max_features=10000, 
                max_length=512, 
                transformer_model='bert-base-uncased',
                use_multiprocessing=True):
        """
        初始化文本预处理器
        
        Args:
            max_features (int): 词汇表大小，用于TF-IDF和CountVectorizer
            max_length (int): 最大序列长度，用于transformer模型
            transformer_model (str): 预训练Transformer模型名称
            use_multiprocessing (bool): 是否使用多进程加速处理
        """
        self.max_features = max_features
        self.max_length = max_length
        self.transformer_model = transformer_model
        self.use_multiprocessing = use_multiprocessing
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        
        # 初始化文本处理工具
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # 并行处理相关
        self.n_jobs = multiprocessing.cpu_count() - 1 if multiprocessing.cpu_count() > 1 else 1
        
        # 如果指定了transformer_model，则加载tokenizer
        if transformer_model:
            self.load_transformer_tokenizer()
        
        logger.info(f"TextPreprocessor初始化完成，最大特征数: {max_features}, Transformer模型: {transformer_model}")
    
    def _parallel_process(self, func, texts):
        """
        使用多进程并行处理文本
        
        Args:
            func: 处理函数
            texts: 文本列表
            
        Returns:
            list: 处理后的文本列表
        """
        if not self.use_multiprocessing or len(texts) < 1000:
            return [func(text) for text in tqdm(texts, desc="处理文本")]
        
        # 使用joblib并行处理
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(func)(text) for text in texts
        )
        return results
    
    def load_transformer_tokenizer(self):
        """
        加载Transformer分词器
        
        Returns:
            AutoTokenizer: Hugging Face分词器
        """
        logger.info(f"加载Transformer分词器: {self.transformer_model}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.transformer_model)
            logger.info(f"成功加载tokenizer: {self.transformer_model}")
        except Exception as e:
            logger.error(f"加载tokenizer失败: {e}")
            self.tokenizer = None

    def basic_clean(self, text):
        """
        基本文本清洗
        
        Args:
            text (str): 输入文本
            
        Returns:
            str: 清洗后的文本
        """
        if not isinstance(text, str):
            logger.warning(f"输入不是字符串类型，尝试转换。类型: {type(text)}")
            try:
                text = str(text)
            except:
                logger.error(f"无法将输入转换为字符串: {text}")
                return ""
        
        # 去除HTML和XML标签
        text = re.sub(r'<.*?>', '', text)
        
        # 去除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 去除无意义的标点符号重复
        text = re.sub(r'([.,!?]){2,}', r'\1', text)
        
        # 清除开头和结尾的空白
        text = text.strip()
        
        return text
    
    def advanced_clean(self, text):
        """
        高级文本清洗：分词、去除停用词、词干提取
        
        Args:
            text (str): 输入文本
            
        Returns:
            str: 清洗后的文本
        """
        if not isinstance(text, str):
            return ""
            
        # 基本清洗
        text = self.basic_clean(text)
        
        # 分词
        tokens = word_tokenize(text)
        
        # 去除停用词
        tokens = [word for word in tokens if word not in self.stop_words]
        
        # 词干提取
        tokens = [self.stemmer.stem(word) for word in tokens]
        
        # 重新组合为文本
        text = ' '.join(tokens)
        
        # 转换为小写
        text = text.lower()
        
        # 移除URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # 移除@用户名
        text = re.sub(r'@\w+', '', text)
        
        # 移除特殊字符和数字
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_texts(self, texts, method='basic'):
        """
        批量预处理文本
        
        Args:
            texts (list): 输入文本列表
            method (str): 预处理方法，可选值: 'basic', 'advanced', 'none'
            
        Returns:
            list: 预处理后的文本列表
        """
        logger.info(f"使用 {method} 方法处理 {len(texts)} 条文本...")
        
        if method == 'none':
            return texts
        
        if method == 'basic':
            cleaned_texts = self._parallel_process(self.basic_clean, texts)
        elif method == 'advanced':
            cleaned_texts = self._parallel_process(self.advanced_clean, texts)
        else:
            logger.warning(f"未知的预处理方法 {method}，使用基本清洗")
            cleaned_texts = self._parallel_process(self.basic_clean, texts)
            
        logger.info(f"文本处理完成，结果示例: {cleaned_texts[0][:100]}...")
        return cleaned_texts
    
    def get_tfidf_features(self, texts, train=True):
        """
        提取TF-IDF特征
        
        Args:
            texts (list): 输入文本列表
            train (bool): 是否在训练模式
            
        Returns:
            scipy.sparse.csr_matrix: TF-IDF特征矩阵
        """
        logger.info(f"提取TF-IDF特征，最大特征数: {self.max_features}...")
        
        if train or self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_features, 
                stop_words='english',
                ngram_range=(1, 2)
            )
            features = self.tfidf_vectorizer.fit_transform(texts)
            logger.info(f"TF-IDF特征提取完成，特征维度: {features.shape}")
        else:
            features = self.tfidf_vectorizer.transform(texts)
            logger.info(f"TF-IDF特征提取完成，特征维度: {features.shape}")
        
        return features
    
    def get_count_features(self, texts, train=True):
        """
        提取词频特征
        
        Args:
            texts (list): 输入文本列表
            train (bool): 是否在训练模式
            
        Returns:
            scipy.sparse.csr_matrix: 词频特征矩阵
        """
        logger.info(f"提取词频特征，最大特征数: {self.max_features}...")
        
        if train or self.count_vectorizer is None:
            self.count_vectorizer = CountVectorizer(
                max_features=self.max_features,
                stop_words='english',
                ngram_range=(1, 2)
            )
            features = self.count_vectorizer.fit_transform(texts)
            logger.info(f"词频特征提取完成，特征维度: {features.shape}")
        else:
            features = self.count_vectorizer.transform(texts)
            logger.info(f"词频特征提取完成，特征维度: {features.shape}")
        
        return features
    
    def get_transformer_features(self, texts, tokenizer=None):
        """
        获取Transformer模型的输入特征
        
        Args:
            texts (list): 输入文本列表
            tokenizer (AutoTokenizer, optional): Transformer分词器
            
        Returns:
            dict: 包含input_ids, attention_mask等的字典
        """
        logger.info(f"提取transformer特征，max_length={self.max_length}")
        
        tokenizer = tokenizer or self.tokenizer
        if tokenizer is None:
            logger.error("Tokenizer未初始化，无法提取transformer特征")
            raise ValueError("Tokenizer未初始化")
            
        # 分批处理，避免内存溢出
        batch_size = 32
        features_list = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_features = tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            features_list.append(batch_features)
            
        # 合并批次结果
        if len(features_list) == 1:
            features = features_list[0]
        else:
            # 根据需要进行合并
            features = {
                'input_ids': np.vstack([f['input_ids'].numpy() for f in features_list]),
                'attention_mask': np.vstack([f['attention_mask'].numpy() for f in features_list])
            }
        
        logger.info(f"Transformer特征提取完成")
        return features
    
    def generate_features(self, X_train, X_test, feature_type='tfidf', clean_method='basic'):
        """
        生成训练集和测试集的特征
        
        Args:
            X_train (list): 训练集文本
            X_test (list): 测试集文本
            feature_type (str): 特征类型，可选值: 'tfidf', 'count', 'transformer', 'combined'
            clean_method (str): 文本清洗方法，可选值: 'basic', 'advanced', 'none'
            
        Returns:
            tuple: (训练集特征, 测试集特征)
        """
        logger.info(f"生成 {feature_type} 特征，使用 {clean_method} 清洗方法...")
        
        # 文本清洗
        X_train_clean = self.preprocess_texts(X_train, method=clean_method)
        X_test_clean = self.preprocess_texts(X_test, method=clean_method)
        
        # 特征提取
        if feature_type == 'tfidf':
            X_train_features = self.get_tfidf_features(X_train_clean, train=True)
            X_test_features = self.get_tfidf_features(X_test_clean, train=False)
            
        elif feature_type == 'count':
            X_train_features = self.get_count_features(X_train_clean, train=True)
            X_test_features = self.get_count_features(X_test_clean, train=False)
            
        elif feature_type == 'transformer':
            X_train_features = self.get_transformer_features(X_train_clean)
            X_test_features = self.get_transformer_features(X_test_clean)
            
        elif feature_type == 'combined':
            # 结合TF-IDF和词频特征
            X_train_tfidf = self.get_tfidf_features(X_train_clean, train=True)
            X_test_tfidf = self.get_tfidf_features(X_test_clean, train=False)
            
            X_train_count = self.get_count_features(X_train_clean, train=True)
            X_test_count = self.get_count_features(X_test_clean, train=False)
            
            # 返回特征列表
            X_train_features = [X_train_tfidf, X_train_count]
            X_test_features = [X_test_tfidf, X_test_count]
            
        else:
            logger.warning(f"未知的特征类型 {feature_type}，使用TF-IDF特征")
            X_train_features = self.get_tfidf_features(X_train_clean, train=True)
            X_test_features = self.get_tfidf_features(X_test_clean, train=False)
        
        logger.info("特征生成完成")
        return X_train_features, X_test_features 