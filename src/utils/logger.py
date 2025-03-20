#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
日志模块 - 用于记录训练过程中的各种信息

提供了多种日志级别和格式，支持同时输出到控制台和文件
可以记录训练过程中的损失、指标、配置等信息
"""

import os
import logging
import sys
import time
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class TrainingLogger:
    """训练日志记录器，用于记录训练过程中的各种信息"""
    
    def __init__(self, log_dir="./logs", experiment_name=None, 
                 console_level=logging.INFO, file_level=logging.DEBUG,
                 enable_tensorboard=False):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志保存目录
            experiment_name: 实验名称，如果为None则使用时间戳
            console_level: 控制台日志级别
            file_level: 文件日志级别
            enable_tensorboard: 是否启用TensorBoard记录
        """
        # 创建日志目录
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置实验名称，默认使用时间戳
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"experiment_{timestamp}"
        else:
            self.experiment_name = experiment_name
            
        # 创建实验日志目录
        self.experiment_dir = self.log_dir / self.experiment_name
        self.experiment_dir.mkdir(exist_ok=True)
        
        # 创建logger实例
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.DEBUG)  # 设置为最低级别，让handler决定哪些被处理
        
        # 清除之前的handlers防止重复
        if self.logger.handlers:
            self.logger.handlers = []
        
        # 添加控制台handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # 添加文件handler
        log_file = self.experiment_dir / f"{self.experiment_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # 训练指标记录
        self.metrics_file = self.experiment_dir / "metrics.json"
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "learning_rate": [],
            "epochs": []
        }
        
        # 记录训练开始时间
        self.start_time = time.time()
        
        # TensorBoard支持
        self.tensorboard_writer = None
        if enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tensorboard_dir = self.experiment_dir / "tensorboard"
                self.tensorboard_writer = SummaryWriter(log_dir=tensorboard_dir)
                self.logger.info(f"TensorBoard日志将保存在 {tensorboard_dir}")
            except ImportError:
                self.logger.warning("未能导入torch.utils.tensorboard.SummaryWriter，TensorBoard记录将被禁用")
        
        # 记录系统信息
        self._log_system_info()
    
    def _log_system_info(self):
        """记录系统和环境信息"""
        import platform
        
        # 获取GPU信息
        gpu_info = "未检测"
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_info = f"可用 ({gpu_count}个设备)"
                for i in range(gpu_count):
                    gpu_info += f"\n  - {torch.cuda.get_device_name(i)}"
            else:
                gpu_info = "不可用"
        except ImportError:
            gpu_info = "PyTorch未安装"
        
        # 获取内存信息
        mem_info = "未检测"
        try:
            import psutil
            mem = psutil.virtual_memory()
            mem_info = f"总计 {mem.total/1024**3:.1f}GB, 可用 {mem.available/1024**3:.1f}GB"
        except ImportError:
            mem_info = "psutil未安装"
        
        # 记录信息
        self.logger.info("-------- 系统信息 --------")
        self.logger.info(f"操作系统: {platform.system()} {platform.release()}")
        self.logger.info(f"Python版本: {platform.python_version()}")
        self.logger.info(f"处理器: {platform.processor()}")
        self.logger.info(f"GPU: {gpu_info}")
        self.logger.info(f"内存: {mem_info}")
        self.logger.info("--------------------------")
    
    def log_config(self, config):
        """记录训练配置"""
        self.logger.info("-------- 训练配置 --------")
        
        # 如果配置是对象，转换为dict
        if not isinstance(config, dict):
            try:
                config_dict = vars(config)
            except:
                self.logger.warning("无法将配置转换为字典，将直接输出")
                self.logger.info(str(config))
                return
        else:
            config_dict = config
        
        # 记录配置到日志
        for key, value in config_dict.items():
            self.logger.info(f"{key}: {value}")
        
        # 保存配置到文件
        config_file = self.experiment_dir / "config.json"
        try:
            # 尝试进行序列化，处理不能直接序列化的对象
            def convert_to_serializable(obj):
                if hasattr(obj, '__dict__'):
                    return obj.__dict__
                elif hasattr(obj, '__str__'):
                    return str(obj)
                else:
                    return repr(obj)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=4, default=convert_to_serializable)
            self.logger.info(f"配置已保存到 {config_file}")
        except Exception as e:
            self.logger.warning(f"保存配置失败: {str(e)}")
        
        self.logger.info("--------------------------")
    
    def log_epoch(self, epoch, epochs, train_loss, val_metrics=None, learning_rate=None):
        """
        记录每个epoch的训练信息
        
        Args:
            epoch: 当前epoch
            epochs: 总epoch数
            train_loss: 训练损失
            val_metrics: 验证集指标字典，包含'loss', 'accuracy', 'precision', 'recall', 'f1'等
            learning_rate: 当前学习率
        """
        # 更新历史记录
        self.history["train_loss"].append(float(train_loss))
        self.history["epochs"].append(int(epoch))
        
        if learning_rate is not None:
            self.history["learning_rate"].append(float(learning_rate))
        
        # 构建日志消息
        msg = f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f}"
        
        # 如果有验证集指标，添加到消息和历史记录中
        if val_metrics is not None:
            for key, value in val_metrics.items():
                if key in self.history:
                    self.history[f"val_{key}"].append(float(value))
                    msg += f" - val_{key}: {value:.4f}"
        
        # 如果提供了学习率，添加到消息中
        if learning_rate is not None:
            msg += f" - lr: {learning_rate:.6f}"
        
        self.logger.info(msg)
        
        # 保存历史记录到文件
        self._save_history()
        
        # 如果启用了TensorBoard，记录指标
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar('Loss/train', train_loss, epoch)
            if val_metrics is not None:
                for key, value in val_metrics.items():
                    self.tensorboard_writer.add_scalar(f'Metrics/val_{key}', value, epoch)
            if learning_rate is not None:
                self.tensorboard_writer.add_scalar('Learning_rate', learning_rate, epoch)
    
    def log_batch(self, epoch, batch, total_batches, loss, metrics=None):
        """
        记录每个batch的训练信息
        
        Args:
            epoch: 当前epoch
            batch: 当前batch
            total_batches: 总batch数
            loss: 当前batch的损失
            metrics: 当前batch的指标字典
        """
        # 构建基本消息
        msg = f"Epoch {epoch} - Batch {batch}/{total_batches} - loss: {loss:.4f}"
        
        # 如果有其他指标，添加到消息中
        if metrics is not None:
            for key, value in metrics.items():
                msg += f" - {key}: {value:.4f}"
        
        self.logger.debug(msg)
    
    def log_eval(self, metrics, dataset_name="测试集"):
        """
        记录评估结果
        
        Args:
            metrics: 评估指标字典
            dataset_name: 数据集名称
        """
        self.logger.info(f"-------- {dataset_name}评估结果 --------")
        for key, value in metrics.items():
            self.logger.info(f"{key}: {value:.4f}")
        self.logger.info("------------------------------")
        
        # 保存评估结果到文件
        eval_file = self.experiment_dir / f"{dataset_name.replace(' ', '_')}_eval.json"
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)
    
    def log_model_summary(self, model):
        """
        记录模型结构摘要
        
        Args:
            model: 模型对象
        """
        self.logger.info("-------- 模型结构 --------")
        self.logger.info(str(model))
        self.logger.info("--------------------------")
        
        # 计算模型参数数量
        try:
            import torch.nn as nn
            if isinstance(model, nn.Module):
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                self.logger.info(f"总参数数量: {total_params:,}")
                self.logger.info(f"可训练参数数量: {trainable_params:,}")
        except (ImportError, AttributeError, Exception) as e:
            self.logger.warning(f"计算模型参数数量失败: {str(e)}")
    
    def log_time(self, name, start_time=None):
        """
        记录时间消耗
        
        Args:
            name: 阶段名称
            start_time: 开始时间，如果为None，则使用当前时间
        
        Returns:
            当前时间，可用于计算下一阶段的时间消耗
        """
        if start_time is None:
            start_time = self.start_time
        
        elapsed = time.time() - start_time
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        time_str = f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"
        
        self.logger.info(f"{name}耗时: {time_str}")
        return time.time()
    
    def log_message(self, msg, level="info"):
        """
        记录自定义消息
        
        Args:
            msg: 消息内容
            level: 日志级别，可选值为 debug, info, warning, error, critical
        """
        level_methods = {
            "debug": self.logger.debug,
            "info": self.logger.info,
            "warning": self.logger.warning,
            "error": self.logger.error,
            "critical": self.logger.critical
        }
        
        if level in level_methods:
            level_methods[level](msg)
        else:
            self.logger.warning(f"未知的日志级别: {level}，使用info级别")
            self.logger.info(msg)
    
    def finish(self):
        """完成日志记录，生成训练报告和可视化图表"""
        # 记录总训练时间
        total_time = time.time() - self.start_time
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        time_str = f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"
        
        self.logger.info(f"训练完成，总耗时: {time_str}")
        
        # 生成训练曲线图
        self._plot_training_curves()
        
        # 关闭TensorBoard writer
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
    
    def _save_history(self):
        """保存训练历史记录到文件"""
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=4)
    
    def _plot_training_curves(self):
        """绘制训练曲线图"""
        if len(self.history["epochs"]) == 0:
            self.logger.warning("没有足够的历史数据来绘制训练曲线")
            return
        
        try:
            # 绘制损失曲线
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.plot(self.history["epochs"], self.history["train_loss"], label="训练损失")
            if "val_loss" in self.history and len(self.history["val_loss"]) > 0:
                plt.plot(self.history["epochs"], self.history["val_loss"], label="验证损失")
            plt.title("损失曲线")
            plt.xlabel("Epoch")
            plt.ylabel("损失")
            plt.legend()
            
            # 绘制指标曲线
            metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
            for i, metric in enumerate(metrics_to_plot):
                val_key = f"val_{metric}"
                if val_key in self.history and len(self.history[val_key]) > 0:
                    plt.subplot(2, 2, i+2)
                    plt.plot(self.history["epochs"], self.history[val_key], label=f"验证{metric}")
                    plt.title(f"{metric.capitalize()}曲线")
                    plt.xlabel("Epoch")
                    plt.ylabel(metric.capitalize())
                    plt.legend()
            
            # 保存图像
            plt.tight_layout()
            curves_file = self.experiment_dir / "training_curves.png"
            plt.savefig(curves_file)
            plt.close()
            
            self.logger.info(f"训练曲线已保存到 {curves_file}")
        except Exception as e:
            self.logger.warning(f"绘制训练曲线失败: {str(e)}")
    
    def get_logger(self):
        """获取原始logger对象"""
        return self.logger

    # 添加常用日志级别方法
    def debug(self, msg):
        """记录debug级别日志"""
        self.logger.debug(msg)
    
    def info(self, msg):
        """记录info级别日志"""
        self.logger.info(msg)
    
    def warning(self, msg):
        """记录warning级别日志"""
        self.logger.warning(msg)
    
    def error(self, msg):
        """记录error级别日志"""
        self.logger.error(msg)
    
    def critical(self, msg):
        """记录critical级别日志"""
        self.logger.critical(msg)


# 创建一个默认的单例logger实例
_default_logger = None

def get_logger(log_dir="./logs", experiment_name=None, 
              console_level=logging.INFO, file_level=logging.DEBUG,
              enable_tensorboard=False):
    """
    获取默认日志记录器实例
    
    如果实例不存在，则创建一个新的实例
    
    Args:
        参数同TrainingLogger.__init__
    
    Returns:
        TrainingLogger实例
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = TrainingLogger(
            log_dir=log_dir,
            experiment_name=experiment_name,
            console_level=console_level,
            file_level=file_level,
            enable_tensorboard=enable_tensorboard
        )
    return _default_logger

# 便捷函数
def set_default_logger(logger):
    """设置默认日志记录器"""
    global _default_logger
    _default_logger = logger

def info(msg):
    """记录info级别日志"""
    logger = get_logger()
    logger.info(msg)

def debug(msg):
    """记录debug级别日志"""
    logger = get_logger()
    logger.debug(msg)

def warning(msg):
    """记录warning级别日志"""
    logger = get_logger()
    logger.warning(msg)

def error(msg):
    """记录error级别日志"""
    logger = get_logger()
    logger.error(msg)

def critical(msg):
    """记录critical级别日志"""
    logger = get_logger()
    logger.critical(msg) 