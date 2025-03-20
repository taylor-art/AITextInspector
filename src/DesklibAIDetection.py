#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：AITextInspector 
@File    ：DesklibAIDetection.py
@Author  ：Swift
@Date    ：2025/3/20 11:29 
'''
import torch
import torch.nn as nn
import logging
import sys
from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel

# 配置日志
def setup_logger():
    # 检查logger是否已存在
    logger = logging.getLogger('AI检测器')
    
    # 如果logger已经有处理器，说明已经初始化过，直接返回
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.DEBUG)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # 创建文件处理器
    file_handler = logging.FileHandler('ai_detector.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        # 在需要的地方获取logger
        logger = setup_logger()
        logger.info("正在初始化AI检测模型")
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()
        logger.debug(f"模型配置信息: {config}")

    def forward(self, input_ids, attention_mask=None, labels=None):
        logger = setup_logger()
        logger.debug(f"前向传播 - 输入形状: {input_ids.shape}")
        
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        logger.debug(f"池化输出形状: {pooled_output.shape}")
        
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())
            logger.debug(f"计算得到的损失值: {loss.item()}")

        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        return output

def predict_single_text(text, model, tokenizer, device, max_len=768, threshold=0.5):
    logger = setup_logger()
    logger.info("开始文本预测")
    logger.debug(f"文本长度: {len(text)} 字符")
    
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    logger.debug(f"编码后的输入形状: {encoded['input_ids'].shape}")
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    logger.debug(f"使用设备: {device}")
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        probability = torch.sigmoid(logits).item()
        
    label = 1 if probability >= threshold else 0
    logger.info(f"预测完成 - AI生成概率: {probability:.4f}, 标签: {'AI生成' if label == 1 else '人工撰写'}")
    return probability, label

def predict_multiple_texts(texts, model, tokenizer, device, max_len=768, threshold=0.5):
    """
    批量预测多个文本
    
    Args:
        texts: 文本列表
        model: AI检测模型
        tokenizer: 分词器
        device: 计算设备
        max_len: 最大文本长度
        threshold: AI生成概率阈值
    
    Returns:
        results: 包含每个文本预测结果的列表
    """
    logger = setup_logger()
    logger.info(f"开始批量预测，共 {len(texts)} 个文本")
    results = []
    
    for i, text in enumerate(texts, 1):
        logger.info(f"正在处理第 {i} 个文本")
        try:
            probability, label = predict_single_text(text, model, tokenizer, device, max_len, threshold)
            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,  # 只显示前100个字符
                'probability': probability,
                'label': label,
                'prediction': 'AI生成' if label == 1 else '人工撰写'
            })
        except Exception as e:
            logger.error(f"处理第 {i} 个文本时发生错误: {str(e)}")
            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'error': str(e)
            })
    
    logger.info("批量预测完成")
    return results

def main():
    logger = setup_logger()
    logger.info("启动AI检测程序")
    
    model_directory = "desklib/ai-text-detector-v1.01"
    logger.info(f"正在从以下位置加载模型: {model_directory}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
        logger.info("分词器加载成功")
        
        model = DesklibAIDetectionModel.from_pretrained(model_directory)
        logger.info("模型加载成功")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")
        model.to(device)
        
        # 测试文本列表
        test_texts = [
            # 英文文本
            "This is a human-written text that contains some personal views and experiences. I went to the supermarket to buy some fruit yesterday. I found it a little expensive.",
            "According to the analysis, the project is expected to be completed in the second quarter of 2024 and will deliver approximately 15% earnings growth at that time. Studies have shown a high degree of confidence in this prediction.",
            "I'm an AI assistant, and I can help you answer questions, write content, provide information, and make recommendations. Just tell me what you need and I'll try to provide you with the most useful response.",
            "Yesterday I had a dream that I was standing on a high mountain, watching the sunrise in the distance. It was a wonderful feeling and reminded me of when I was a kid and I used to travel with my family.",
            
            # 中文新闻类文本
            "据新华社报道，近日，国务院常务会议审议通过《关于促进人工智能产业健康发展的指导意见》。会议指出，要坚持创新引领，完善产业链供应链，提升产业基础能力，推动人工智能与实体经济深度融合，促进产业转型升级和高质量发展。",
            
            # 中文日常对话文本
            "昨天我和朋友去看了一场电影，真的特别好看！剧情紧凑，演员演技在线，特效也很震撼。看完后我们还去附近的咖啡馆聊了很久，分享了各自最近的生活。这样的周末真的很充实。",
            
            # 中文技术文档类文本
            "本项目采用Python3.8开发，主要依赖包括TensorFlow 2.4.0和PyTorch 1.8.0。在开始开发之前，请确保已经正确配置了开发环境，并安装了所有必要的依赖包。项目的主要功能模块包括数据预处理、模型训练和预测评估三个部分。",
            
            # 中文学术类文本
            "基于深度学习的自然语言处理技术在近年来取得了显著进展。通过预训练语言模型和迁移学习方法，模型能够更好地理解和生成人类语言。本研究通过实验证明，在特定领域数据上进行微调可以显著提升模型性能。",
            
            # 中文广告文案
            "限时特惠！全场商品五折起，更有满1000减200优惠券等你来拿！新款春装已经上架，时尚潮流，品质保证。快来挑选心仪的商品吧！",
            
            # 中文诗歌类文本
            "春风拂过山野，带来泥土的芬芳。远处的山峦如黛，近处的溪流潺潺。我站在这里，聆听大自然的声音，感受生命的律动。",
            
            # 中文AI生成文本
            "作为一个AI助手，我可以帮助您完成各种任务，包括文本创作、数据分析、问题解答等。我的知识库涵盖多个领域，可以为您提供准确和有见地的回答。请告诉我您需要什么帮助。",
            
            # 中文专业术语文本
            "量子计算利用量子比特的叠加态和纠缠效应，可以在特定问题上实现指数级的计算加速。目前，超导量子计算机已经实现了量子优越性的初步验证。",
            
            # 中文情感类文本
            "今天是我们结婚十周年纪念日，回想这十年的点点滴滴，有欢笑也有泪水，但更多的是温暖和感动。感谢一路相伴，愿未来继续携手同行。",
            
            # 中文公文类文本
            "经研究决定，即日起在全市范围内开展为期三个月的城市环境综合整治行动。各区街道办事处要高度重视，细化工作方案，落实工作责任，确保整治工作取得实效。"
        ]
        
        # 批量预测
        results = predict_multiple_texts(test_texts, model, tokenizer, device)
        
        # 打印结果
        print("\n=== 预测结果 ===")
        for i, result in enumerate(results, 1):
            print(f"\n文本 {i}:")
            print(f"内容预览: {result['text']}")
            if 'error' in result:
                print(f"错误: {result['error']}")
            else:
                print(f"AI生成概率: {result['probability']:.4f}")
                print(f"预测结果: {result['prediction']}")
                print(f"文本类型: {'中文' if any(ord(c) > 127 for c in result['text']) else '英文'}")
        print("\n=== 预测完成 ===")
        
    except Exception as e:
        logger.error(f"发生错误: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
