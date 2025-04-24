#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torchaudio
import struct
from typing import Optional, List, Tuple

# 导入Whisper模型用于ASR
import whisper


class AudioProcessor:
    """音频处理类，负责音频格式转换和语音识别"""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        
        # 初始化Whisper模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.asr_model = whisper.load_model("medium", device=self.device)
        
        print(f"音频处理器初始化完成，使用设备: {self.device}")
        
    def bytes_to_audio(self, audio_bytes: bytes) -> np.ndarray:
        """将字节数据转换为浮点音频数据"""
        # 假设输入是float32格式的音频数据
        floats_count = len(audio_bytes) // 4  # 4字节表示一个float
        audio_data = struct.unpack(f'{floats_count}f', audio_bytes)
        return np.array(audio_data, dtype=np.float32)
    
    def audio_to_bytes(self, audio_data: np.ndarray) -> bytes:
        """将浮点音频数据转换为字节数据"""
        return struct.pack(f'{len(audio_data)}f', *audio_data)
    
    def normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """归一化音频数据"""
        if audio_data.size == 0:
            return audio_data
            
        # 归一化到 [-1, 1] 范围
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val
        return audio_data
    
    def resample_audio(self, audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """重采样音频"""
        if orig_sr == target_sr:
            return audio_data
            
        # 转换为tensor以使用torchaudio进行重采样
        waveform = torch.tensor(audio_data).unsqueeze(0)  # [1, time]
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        resampled_waveform = resampler(waveform)
        
        return resampled_waveform.squeeze(0).numpy()
    
    def recognize_speech(self, audio_data: np.ndarray) -> str:
        """使用Whisper模型进行语音识别"""
        if audio_data.size == 0:
            return ""
            
        try:
            # 归一化音频
            audio_data = self.normalize_audio(audio_data)
            
            # 确保音频的采样率与Whisper模型所需一致（Whisper要求16kHz）
            if self.sample_rate != 16000:
                audio_data = self.resample_audio(audio_data, self.sample_rate, 16000)
            
            # 转换为Whisper需要的格式
            audio_tensor = torch.tensor(audio_data)
            
            # 执行语音识别
            result = self.asr_model.transcribe(
                audio_tensor.numpy(), 
                language="zh",  # 中文语音识别
                fp16=torch.cuda.is_available()  # 如果有GPU则使用fp16加速
            )
            
            # 返回识别结果
            return result["text"].strip()
            
        except Exception as e:
            print(f"语音识别失败: {e}")
            return "" 