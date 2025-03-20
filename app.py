#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from waitress import serve
from src.app.web_app import create_app
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='启动AI文本检测Web应用')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--model_type', type=str, required=True, 
                        choices=['logistic', 'random_forest', 'svm', 'gradient_boosting', 'bert', 'roberta'],
                        help='模型类型')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='主机地址')
    parser.add_argument('--port', type=int, default=5000, help='端口号')
    parser.add_argument('--debug', action='store_true', help='是否启用调试模式')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 确保模型文件存在
    if not os.path.exists(args.model_path):
        logger.error(f"模型文件不存在: {args.model_path}")
        return
    
    # 创建Flask应用
    app = create_app(model_path=args.model_path, model_name=args.model_type)
    
    # 启动应用
    logger.info(f"启动Web应用，监听地址: {args.host}:{args.port}")
    
    if args.debug:
        # 开发模式
        app.run(host=args.host, port=args.port, debug=True)
    else:
        # 生产模式使用waitress
        serve(app, host=args.host, port=args.port)
    
if __name__ == '__main__':
    main() 