import torch
import logging
import os
import numpy as np
from ..data_processing.preprocessor import TextPreprocessor
from ..models.bert_model import AITextDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AITextDetectorApp:
    """
    AI文本检测应用类，用于实际检测文本
    """
    def __init__(self, model_path, model_name="desklib/ai-text-detector-v1.01", device=None):
        """
        初始化检测器应用
        
        Args:
            model_path: 模型路径
            model_name: 预训练模型名称
            device: 运行设备 ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载预处理器
        logger.info(f"初始化预处理器: {model_name}")
        self.preprocessor = TextPreprocessor(model_name=model_name)
        
        # 加载模型
        logger.info(f"从路径加载模型: {model_path}")
        try:
            self.model = AITextDetector.load(model_path, model_name=model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def detect(self, text, threshold=0.5):
        """
        检测文本是否为AI生成
        
        Args:
            text: 待检测文本
            threshold: 阈值，大于阈值判断为AI生成
            
        Returns:
            dict: 检测结果，包含'is_ai_generated'和'confidence'
        """
        # 文本预处理
        encoded = self.preprocessor.tokenize(text)
        
        # 将输入移至设备
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # 预测
        with torch.no_grad():
            _, logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        # 判断结果
        ai_prob = probs[0, 1]  # AI生成的概率
        is_ai = ai_prob > threshold
        
        result = {
            'is_ai_generated': bool(is_ai),
            'confidence': float(ai_prob)
        }
        
        logger.info(f"检测结果: {'AI生成' if is_ai else '人类撰写'}, 置信度: {ai_prob:.4f}")
        
        return result
    
    def detect_file(self, file_path, threshold=0.5):
        """
        检测文件内容是否为AI生成
        
        Args:
            file_path: 文件路径
            threshold: 阈值，大于阈值判断为AI生成
            
        Returns:
            dict: 检测结果，包含'is_ai_generated'和'confidence'
        """
        logger.info(f"检测文件: {file_path}")
        
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # 检测文本
            result = self.detect(text, threshold)
            
            return result
        except Exception as e:
            logger.error(f"检测文件失败: {e}")
            raise
    
    def batch_detect(self, texts, threshold=0.5):
        """
        批量检测文本
        
        Args:
            texts: 文本列表
            threshold: 阈值，大于阈值判断为AI生成
            
        Returns:
            list: 检测结果列表，每项包含'is_ai_generated'和'confidence'
        """
        logger.info(f"批量检测{len(texts)}个文本")
        
        # 文本预处理
        encoded = self.preprocessor.tokenize(texts)
        
        # 将输入移至设备
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # 预测
        with torch.no_grad():
            _, logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        # 处理结果
        results = []
        for i, prob in enumerate(probs):
            ai_prob = prob[1]  # AI生成的概率
            is_ai = ai_prob > threshold
            
            result = {
                'is_ai_generated': bool(is_ai),
                'confidence': float(ai_prob)
            }
            
            results.append(result)
        
        logger.info(f"批量检测完成，AI生成文本数量: {sum(r['is_ai_generated'] for r in results)}")
        
        return results 