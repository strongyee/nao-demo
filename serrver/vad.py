#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from typing import List, Optional

class SileroVAD:
    """使用Silero VAD进行语音活动检测"""
    
    def __init__(
        self, 
        threshold: float = 0.5, 
        sampling_rate: int = 16000, 
        window_size_samples: int = 512,
        device: str = None
    ):
        # 如果未指定设备，则自动选择
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.window_size_samples = window_size_samples
        
        # 加载Silero VAD模型
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
            verbose=False
        )
        
        # 获取模型函数
        self.model = model.to(self.device)
        self.get_speech_timestamps = utils[0]
        self.get_speech_ts_adaptive = utils[4]  # 自适应阈值
        
        print(f"Silero VAD初始化完成，使用设备: {self.device}")
    
    def process_audio(self, audio_data: np.ndarray) -> List[np.ndarray]:
        """
        处理音频数据，检测语音活动并返回包含语音的音频段
        
        Args:
            audio_data: 输入音频数据，单声道浮点数组
            
        Returns:
            包含检测到的语音段的列表
        """
        if audio_data.size == 0:
            return []
            
        # 确保音频数据在正确的范围内 [-1, 1]
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
            
        # 转换为PyTorch张量
        audio_tensor = torch.tensor(audio_data).to(self.device)
        
        # 获取语音时间戳
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor, 
            self.model,
            threshold=self.threshold,
            sampling_rate=self.sampling_rate,
            window_size_samples=self.window_size_samples,
            min_speech_duration_ms=100,  # 最小语音持续时间(毫秒)
            min_silence_duration_ms=100  # 最小静音持续时间(毫秒)
        )
        
        # 提取语音段
        speech_segments = []
        for ts in speech_timestamps:
            segment = audio_data[ts['start']:ts['end']]
            speech_segments.append(segment)
            
        return speech_segments
    
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        判断一小段音频是否包含语音
        
        Args:
            audio_chunk: 音频片段
            
        Returns:
            布尔值，指示是否检测到语音
        """
        if audio_chunk.size == 0:
            return False
            
        # 确保音频数据在正确的范围内
        if np.max(np.abs(audio_chunk)) > 1.0:
            audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))
            
        # 转换为PyTorch张量
        audio_tensor = torch.tensor(audio_chunk).to(self.device)
        
        # 使用模型进行预测
        with torch.no_grad():
            speech_prob = self.model(audio_tensor, self.sampling_rate).item()
            
        # 根据阈值判断是否是语音
        return speech_prob >= self.threshold 