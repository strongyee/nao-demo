#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from typing import Optional

# 尝试导入Orpheus TTS
try:
    from orpheus import OrpheusEngine
except ImportError:
    print("警告: 无法导入Orpheus库，请安装: pip install orpheus-tts")


class OrpheusTTS:
    """
    Orpheus TTS语音合成类
    """
    
    def __init__(self, model_path: str = "models/orpheus", device: str = None):
        """
        初始化Orpheus TTS引擎
        
        Args:
            model_path: Orpheus模型路径或模型名称
            device: 设备 ("cuda"或"cpu")
        """
        # 如果未指定设备，则自动选择
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model_path = model_path
        
        # 创建模型目录
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # 初始化Orpheus引擎
        try:
            self.engine = OrpheusEngine(
                model=model_path,
                device=self.device
            )
            print(f"Orpheus TTS引擎初始化完成，使用设备: {self.device}")
        except Exception as e:
            print(f"Orpheus TTS引擎初始化失败: {e}")
            self.engine = None
    
    def synthesize(self, text: str) -> Optional[bytes]:
        """
        将文本合成为语音
        
        Args:
            text: 要合成的文本
            
        Returns:
            音频数据字节，如果失败则返回None
        """
        if self.engine is None:
            print("TTS引擎未初始化")
            return None
            
        if not text:
            return None
            
        try:
            # 使用Orpheus生成语音
            audio_array = self.engine.synthesize(
                text=text,
                language='zh', # 中文
                reference_speaker=None, # 使用默认说话人
                speed=1.0,     # 正常速度
                emotion=None   # 默认情感
            )
            
            # 转换为16位PCM
            audio_int16 = (audio_array * 32767).astype(np.int16)
            
            # 转换为字节
            audio_bytes = audio_int16.tobytes()
            
            return audio_bytes
            
        except Exception as e:
            print(f"语音合成失败: {e}")
            return None
    
    def available_languages(self) -> list:
        """获取可用的语言列表"""
        if self.engine:
            return self.engine.available_languages
        return []
    
    def available_speakers(self) -> list:
        """获取可用的说话人列表"""
        if self.engine:
            return self.engine.available_speakers
        return []

# TTS后备方案：使用pyttsx3
class FallbackTTS:
    """当Orpheus不可用时的后备TTS引擎"""
    
    def __init__(self):
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 180)  # 语速
            self.engine.setProperty('volume', 0.9)  # 音量
            
            # 尝试设置中文语音
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if 'chinese' in voice.languages[0].lower() or 'zh' in voice.id.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
            
            print("后备TTS引擎初始化完成")
        except Exception as e:
            print(f"后备TTS引擎初始化失败: {e}")
            self.engine = None
    
    def synthesize(self, text: str) -> Optional[bytes]:
        if self.engine is None:
            return None
            
        try:
            # 将音频保存到临时文件
            import tempfile
            import wave
            import array
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            self.engine.save_to_file(text, temp_path)
            self.engine.runAndWait()
            
            # 读取音频文件
            with wave.open(temp_path, 'rb') as wave_file:
                frames = wave_file.readframes(wave_file.getnframes())
            
            # 删除临时文件
            os.remove(temp_path)
            
            return frames
            
        except Exception as e:
            print(f"后备TTS合成失败: {e}")
            return None 