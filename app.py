#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：AITextInspector 
@File    ：app.py
@Author  ：Swift
@Date    ：2025/3/20 11:29 
'''

import os
import torch
import json
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer
from src.DesklibAIDetection import DesklibAIDetectionModel, predict_single_text, setup_logger

# 直接使用更新后的logger
logger = setup_logger()

app = Flask(__name__)

# 全局变量用于存储模型和分词器
model = None
tokenizer = None
device = None

def load_model():
    """
    加载AI检测模型
    """
    global model, tokenizer, device
    
    if model is not None:
        return
    
    logger.info("正在加载AI文本检测模型...")
    
    # 直接使用desklib模型
    model_path = "desklib/ai-text-detector-v1.01"
    
    try:
        # 加载分词器和模型
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = DesklibAIDetectionModel.from_pretrained(model_path)
        
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        logger.info(f"模型加载成功，使用设备: {device}")
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}", exc_info=True)
        raise

@app.route('/')
def index():
    """
    渲染主页
    """
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """
    处理文本检测请求
    """
    try:
        # 获取请求数据
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "未提供文本"}), 400
        
        text = data['text']
        if not text.strip():
            return jsonify({"error": "文本内容为空"}), 400
        
        logger.info(f"收到检测请求，文本长度: {len(text)} 字符")
        
        # 确保模型已加载
        if model is None:
            load_model()
        
        # 执行文本检测
        max_len = 768  # 默认最大长度
        threshold = 0.5  # 默认阈值
        
        # 如果请求中包含这些参数，则使用请求中的值
        if 'max_len' in data and isinstance(data['max_len'], int):
            max_len = min(data['max_len'], 1024)  # 限制最大长度
        
        if 'threshold' in data and isinstance(data['threshold'], float):
            threshold = max(0.0, min(data['threshold'], 1.0))  # 限制阈值在0-1之间
        
        # 预测
        probability, label = predict_single_text(
            text, model, tokenizer, device, max_len, threshold
        )
        
        result = {
            "probability": probability,
            "label": label,
            "prediction": "AI生成" if label == 1 else "人工撰写"
        }
        
        logger.info(f"检测完成: {result}")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"检测过程中发生错误: {str(e)}", exc_info=True)
        return jsonify({"error": f"检测失败: {str(e)}"}), 500

@app.route('/status', methods=['GET'])
def status():
    """
    返回系统状态
    """
    return jsonify({
        "status": "running",
        "model_loaded": model is not None,
        "device": str(device) if device else None
    })

if __name__ == "__main__":
    # 启动前预加载模型
    try:
        load_model()
    except Exception as e:
        logger.error(f"预加载模型失败: {str(e)}")
        print(f"警告: 模型预加载失败，将在首次请求时尝试加载。错误: {str(e)}")
    
    # 在生产环境中，应该使用更健壮的WSGI服务器如waitress
    from waitress import serve
    
    # 使用8008端口
    port = 8008
    print("AI文本检测系统启动中...")
    print(f"访问 http://127.0.0.1:{port} 使用系统")
    serve(app, host="127.0.0.1", port=port)
