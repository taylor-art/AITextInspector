import re
import logging
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    文本预处理类，用于清洗和准备文本数据
    """
    def __init__(self, model_name="desklib/ai-text-detector-v1.01", max_length=512):
        """
        初始化TextPreprocessor
        
        Args:
            model_name (str): 使用的预训练模型名称
            max_length (int): 文本最大长度
        """
        self.max_length = max_length
        logger.info(f"加载tokenizer: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"成功加载tokenizer: {model_name}")
        except Exception as e:
            logger.error(f"加载tokenizer失败: {e}")
            raise
            
    def clean_text(self, text):
        """
        清洗文本
        
        Args:
            text (str): 原始文本
            
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
        
        # 去除其他特殊字符
        text = re.sub(r'[^\w\s.,!?;:\-\'\"()]', '', text)
        
        # 清除开头和结尾的空白
        text = text.strip()
        
        return text
        
    def tokenize(self, texts, return_tensors="pt", padding=True, truncation=True):
        """
        对文本进行分词处理
        
        Args:
            texts (list or str): 输入文本或文本列表
            return_tensors (str): 返回张量类型 ('pt': PyTorch)
            padding (bool): 是否对序列进行填充
            truncation (bool): 是否截断超长序列
            
        Returns:
            dict: 包含input_ids, attention_mask等的字典
        """
        # 确保texts是列表
        if isinstance(texts, str):
            texts = [texts]
            
        # 清洗文本
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # 分词
        logger.info(f"对{len(cleaned_texts)}条文本进行分词")
        encoded = self.tokenizer(
            cleaned_texts,
            max_length=self.max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )
        
        logger.info(f"分词完成，得到{len(encoded['input_ids'])}条输入序列")
        return encoded
        
    def prepare_dataset(self, texts, labels=None):
        """
        准备模型输入数据集
        
        Args:
            texts (list): 文本列表
            labels (list, optional): 标签列表
            
        Returns:
            dict: 包含input_ids, attention_mask和labels的字典
        """
        encoded = self.tokenize(texts)
        
        if labels is not None:
            encoded['labels'] = labels
            
        return encoded 