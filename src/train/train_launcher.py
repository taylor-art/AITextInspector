#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI文本检测系统训练启动器
提供多种预设训练配置，可以通过IDE直接启动训练
"""

import argparse
import os
from train import train, parse_args

class TrainingConfigs:
    """预设的训练配置"""
    
    @staticmethod
    def get_quick_test_config():
        """快速测试配置"""
        args = parse_args(['--data_dir', './dataset',
                         '--model_type', 'bert',
                         '--pretrained_model', 'bert-base-uncased',
                         '--max_samples', '10000',
                         '--epochs', '1',
                         '--analyze_data',
                         '--cpu_optimize',
                         '--use_lightweight_model'])
        return args
    
    @staticmethod
    def get_gpu_training_config():
        """GPU训练配置"""
        args = parse_args(['--data_dir', './dataset',
                         '--model_type', 'bert',
                         '--pretrained_model', 'bert-base-uncased',
                         '--batch_size', '16',
                         '--epochs', '3'])
        return args
    
    @staticmethod
    def get_cpu_optimized_config():
        """CPU优化训练配置"""
        args = parse_args(['--data_dir', './dataset',
                         '--model_type', 'bert',
                         '--cpu_optimize',
                         '--use_lightweight_model',
                         '--batch_size', '4',
                         '--max_length', '128'])
        return args
    
    @staticmethod
    def get_random_forest_config():
        """随机森林训练配置"""
        args = parse_args(['--data_dir', './dataset',
                         '--model_type', 'random_forest',
                         '--n_estimators', '100',
                         '--max_depth', '10'])
        return args
    
    @staticmethod
    def get_production_config():
        """生产环境完整训练配置"""
        args = parse_args(['--data_dir', './dataset',
                         '--data_type', 'combined',
                         '--model_type', 'bert',
                         '--pretrained_model', 'bert-base-uncased',
                         '--batch_size', '16',
                         '--epochs', '5',
                         '--max_length', '512',
                         '--feature_type', 'transformer',
                         '--clean_method', 'advanced',
                         '--early_stopping',
                         '--patience', '3',
                         '--dropout_rate', '0.2',
                         '--learning_rate', '2e-5',
                         '--output_dir', './output',
                         '--analyze_data',
                         '--balanced', 'True'])
        return args
    
    @staticmethod
    def get_model_comparison_configs():
        """模型对比实验配置"""
        configs = []
        
        # BERT模型
        bert_args = parse_args(['--model_type', 'bert',
                              '--data_type', 'combined'])
        configs.append(('BERT', bert_args))
        
        # 随机森林
        rf_args = parse_args(['--model_type', 'random_forest',
                            '--data_type', 'combined'])
        configs.append(('RandomForest', rf_args))
        
        # 逻辑回归
        lr_args = parse_args(['--model_type', 'logistic',
                            '--data_type', 'combined'])
        configs.append(('LogisticRegression', lr_args))
        
        return configs
    
    @staticmethod
    def get_feature_comparison_configs():
        """特征提取方法对比配置"""
        configs = []
        
        # TF-IDF特征
        tfidf_args = parse_args(['--feature_type', 'tfidf'])
        configs.append(('TF-IDF', tfidf_args))
        
        # Transformer特征
        transformer_args = parse_args(['--feature_type', 'transformer'])
        configs.append(('Transformer', transformer_args))
        
        # 组合特征
        combined_args = parse_args(['--feature_type', 'combined'])
        configs.append(('Combined', combined_args))
        
        return configs
    
    @staticmethod
    def get_cleaning_comparison_configs():
        """数据清洗方法对比配置"""
        configs = []
        
        # 基础清洗
        basic_args = parse_args(['--clean_method', 'basic'])
        configs.append(('Basic', basic_args))
        
        # 高级清洗
        advanced_args = parse_args(['--clean_method', 'advanced'])
        configs.append(('Advanced', advanced_args))
        
        # 无清洗
        none_args = parse_args(['--clean_method', 'none'])
        configs.append(('None', none_args))
        
        return configs

def run_single_training(config_name, args):
    """运行单个训练配置"""
    print(f"\n{'='*50}")
    print(f"开始训练: {config_name}")
    print(f"{'='*50}")
    
    metrics, model_path = train(args)
    
    print(f"\n训练完成: {config_name}")
    print(f"模型保存路径: {model_path}")
    print(f"性能指标: {metrics}")
    print(f"{'='*50}\n")
    
    return metrics, model_path

def run_comparison_experiment(configs, experiment_name):
    """运行对比实验"""
    print(f"\n{'='*50}")
    print(f"开始{experiment_name}对比实验")
    print(f"{'='*50}")
    
    results = {}
    for config_name, args in configs:
        metrics, model_path = run_single_training(config_name, args)
        results[config_name] = {
            'metrics': metrics,
            'model_path': model_path
        }
    
    # 保存实验结果
    import json
    output_dir = './output/experiments'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, f'{experiment_name}_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def main():
    """主函数：选择要运行的训练配置"""
    print("AI文本检测系统训练启动器")
    print("\n可用的训练配置：")
    print("1. 快速测试")
    print("2. GPU训练")
    print("3. CPU优化训练")
    print("4. 随机森林训练")
    print("5. 生产环境完整训练")
    print("6. 模型对比实验")
    print("7. 特征提取方法对比实验")
    print("8. 数据清洗方法对比实验")
    
    choice = input("\n请选择训练配置（输入数字）: ")
    
    if choice == '1':
        args = TrainingConfigs.get_quick_test_config()
        run_single_training("快速测试", args)
    
    elif choice == '2':
        args = TrainingConfigs.get_gpu_training_config()
        run_single_training("GPU训练", args)
    
    elif choice == '3':
        args = TrainingConfigs.get_cpu_optimized_config()
        run_single_training("CPU优化训练", args)
    
    elif choice == '4':
        args = TrainingConfigs.get_random_forest_config()
        run_single_training("随机森林训练", args)
    
    elif choice == '5':
        args = TrainingConfigs.get_production_config()
        run_single_training("生产环境完整训练", args)
    
    elif choice == '6':
        configs = TrainingConfigs.get_model_comparison_configs()
        run_comparison_experiment(configs, "model_comparison")
    
    elif choice == '7':
        configs = TrainingConfigs.get_feature_comparison_configs()
        run_comparison_experiment(configs, "feature_comparison")
    
    elif choice == '8':
        configs = TrainingConfigs.get_cleaning_comparison_configs()
        run_comparison_experiment(configs, "cleaning_comparison")
    
    else:
        print("无效的选择！")

if __name__ == '__main__':
    main() 