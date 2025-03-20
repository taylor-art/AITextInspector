#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI文本检测命令行工具
"""

import os
import argparse
import logging
import sys
import json

import numpy as np

from src.models.model_factory import ModelFactory
from src.data_processing.preprocess import TextPreprocessor

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_text(text, model_path, model_type, feature_type='tfidf', clean_method='basic'):
    """
    检测文本是否由AI生成
    
    Args:
        text (str): 待检测文本
        model_path (str): 模型路径
        model_type (str): 模型类型
        feature_type (str): 特征类型
        clean_method (str): 文本清洗方法
        
    Returns:
        dict: 检测结果
    """
    logger.info("加载模型...")
    model_factory = ModelFactory()
    model = model_factory.load_model(model_path, model_type)
    
    logger.info("预处理文本...")
    preprocessor = TextPreprocessor(
        transformer_model='bert-base-uncased' if model_type in ['bert', 'roberta'] else None
    )
    
    # 处理文本
    processed_text = preprocessor.preprocess_texts([text], method=clean_method)[0]
    logger.info(f"特征提取方式: {feature_type}")
    
    # 特征提取
    if feature_type == 'tfidf':
        features = preprocessor.get_tfidf_features([processed_text], train=False)
    elif feature_type == 'count':
        features = preprocessor.get_count_features([processed_text], train=False)
    elif feature_type == 'transformer':
        features = preprocessor.get_transformer_features([processed_text])
    else:
        logger.warning(f"不支持的特征类型 {feature_type}，使用TF-IDF")
        features = preprocessor.get_tfidf_features([processed_text], train=False)
    
    # 预测
    logger.info("进行预测...")
    
    if model_type in ['bert', 'roberta']:
        # Transformer模型预测
        prediction_raw = model.predict(features)
        prediction = int(prediction_raw.argmax(axis=1)[0])
        probability = float(prediction_raw[0, 1])
    else:
        # 机器学习模型预测
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0, 1])
    
    result = {
        "text": text[:100] + ("..." if len(text) > 100 else ""),
        "prediction": prediction,
        "prediction_label": "AI生成" if prediction == 1 else "人类撰写",
        "ai_probability": probability,
        "human_probability": 1 - probability
    }
    
    logger.info(f"预测结果: {result['prediction_label']}")
    logger.info(f"AI生成概率: {result['ai_probability']:.4f}")
    
    return result

def detect_file(file_path, model_path, model_type, feature_type='tfidf', clean_method='basic', output=None):
    """
    检测文件中的文本
    
    Args:
        file_path (str): 文本文件路径
        model_path (str): 模型路径
        model_type (str): 模型类型
        feature_type (str): 特征类型
        clean_method (str): 文本清洗方法
        output (str): 输出文件路径
        
    Returns:
        dict: 检测结果
    """
    # 读取文件
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        logger.error(f"读取文件出错: {e}")
        return {"error": f"读取文件出错: {e}"}
    
    # 检测
    result = detect_text(text, model_path, model_type, feature_type, clean_method)
    
    # 添加文件信息
    result["file_path"] = file_path
    
    # 输出结果
    if output:
        try:
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"检测结果已保存到: {output}")
        except Exception as e:
            logger.error(f"保存结果出错: {e}")
    
    return result

def detect_csv(file_path, model_path, model_type, feature_type='tfidf', clean_method='basic', 
               column='abstract', output=None, max_samples=None):
    """
    检测CSV文件中的文本
    
    Args:
        file_path (str): CSV文件路径
        model_path (str): 模型路径
        model_type (str): 模型类型
        feature_type (str): 特征类型
        clean_method (str): 文本清洗方法
        column (str): 包含文本的列名
        output (str): 输出文件路径
        max_samples (int): 最大处理样本数
        
    Returns:
        dict: 检测结果
    """
    import pandas as pd
    
    logger.info(f"读取CSV文件: {file_path}")
    try:
        # 读取CSV文件
        if max_samples:
            df = pd.read_csv(file_path, nrows=max_samples)
        else:
            df = pd.read_csv(file_path)
        
        # 检查列名是否存在
        if column not in df.columns:
            available_columns = ", ".join(df.columns)
            error_msg = f"列 '{column}' 不存在。可用列: {available_columns}"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # 加载模型
        logger.info("加载模型...")
        model_factory = ModelFactory()
        model = model_factory.load_model(model_path, model_type)
        
        # 预处理器
        logger.info("初始化预处理器...")
        preprocessor = TextPreprocessor(
            transformer_model='bert-base-uncased' if model_type in ['bert', 'roberta'] else None
        )
        
        # 处理文本
        logger.info("预处理文本...")
        texts = df[column].astype(str).tolist()
        processed_texts = preprocessor.preprocess_texts(texts, method=clean_method)
        
        # 特征提取
        logger.info(f"特征提取: {feature_type}")
        if feature_type == 'tfidf':
            features = preprocessor.get_tfidf_features(processed_texts, train=True)
        elif feature_type == 'count':
            features = preprocessor.get_count_features(processed_texts, train=True)
        elif feature_type == 'transformer':
            features = preprocessor.get_transformer_features(processed_texts)
        else:
            logger.warning(f"不支持的特征类型 {feature_type}，使用TF-IDF")
            features = preprocessor.get_tfidf_features(processed_texts, train=True)
        
        # 预测
        logger.info("进行预测...")
        if model_type in ['bert', 'roberta']:
            # Transformer模型预测
            predictions_raw = model.predict(features)
            predictions = np.argmax(predictions_raw, axis=1)
            probabilities = predictions_raw[:, 1]
        else:
            # 机器学习模型预测
            predictions = model.predict(features)
            probabilities = model.predict_proba(features)[:, 1]
        
        # 创建结果
        results = []
        for i, (text, pred, prob) in enumerate(zip(texts, predictions, probabilities)):
            result = {
                "id": i,
                "text_preview": text[:100] + ("..." if len(text) > 100 else ""),
                "prediction": int(pred),
                "prediction_label": "AI生成" if pred == 1 else "人类撰写",
                "ai_probability": float(prob),
                "human_probability": 1 - float(prob)
            }
            results.append(result)
        
        # 汇总
        summary = {
            "total": len(results),
            "ai_generated": int(sum(r["prediction"] == 1 for r in results)),
            "human_written": int(sum(r["prediction"] == 0 for r in results)),
            "ai_percentage": float(sum(r["prediction"] == 1 for r in results) / len(results) * 100),
            "results": results
        }
        
        # 输出结果
        if output:
            try:
                with open(output, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                logger.info(f"检测结果已保存到: {output}")
            except Exception as e:
                logger.error(f"保存结果出错: {e}")
        
        return summary
    
    except Exception as e:
        logger.error(f"处理CSV文件时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": f"处理CSV文件时出错: {str(e)}"}

def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description='AI文本检测命令行工具')
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', type=str, help='待检测文本')
    input_group.add_argument('--file', type=str, help='待检测文本文件路径')
    input_group.add_argument('--csv', type=str, help='待检测CSV文件路径')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型路径')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['logistic', 'random_forest', 'svm', 'gradient_boosting', 'bert', 'roberta'],
                       help='模型类型')
    parser.add_argument('--feature_type', type=str, default='tfidf',
                       choices=['tfidf', 'count', 'transformer'],
                       help='特征类型')
    parser.add_argument('--clean_method', type=str, default='basic',
                       choices=['basic', 'advanced', 'none'],
                       help='文本清洗方法')
    parser.add_argument('--column', type=str, default='abstract',
                       help='CSV文件中包含文本的列名')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='CSV文件中最大处理样本数')
    parser.add_argument('--output', type=str, help='输出结果文件路径')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    try:
        if args.text:
            result = detect_text(
                args.text, 
                args.model_path, 
                args.model_type,
                args.feature_type,
                args.clean_method
            )
        elif args.file:
            result = detect_file(
                args.file,
                args.model_path,
                args.model_type,
                args.feature_type,
                args.clean_method,
                args.output
            )
        elif args.csv:
            result = detect_csv(
                args.csv,
                args.model_path,
                args.model_type,
                args.feature_type,
                args.clean_method,
                args.column,
                args.output,
                args.max_samples
            )
        
        # 打印结果
        if not args.output:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
    except Exception as e:
        logger.error(f"检测过程出错: {e}")
        sys.exit(1) 