#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from typing import List, Dict, Any, Optional

# 导入Qwen相关库
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenChat:
    """
    使用Qwen 2.5模型的聊天接口
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5:1.5b", device: str = None):
        """
        初始化Qwen聊天模型
        
        Args:
            model_name: 模型名称或路径
            device: 设备 ("cuda", "cpu", "cuda:0", etc.)
        """
        # 如果未指定设备，则自动选择
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model_name = model_name
        
        # 为8位量化和闪存注意力启用优化
        quantization = device.startswith("cuda")
        
        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
        
        # 加载模型和分词器
        try:
            print(f"正在加载Qwen模型: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                quantization_config={"load_in_8bit": quantization}
            )
            
            print(f"Qwen模型加载完成，使用设备: {self.device}")
        except Exception as e:
            print(f"加载Qwen模型失败: {e}")
            self.model = None
            self.tokenizer = None
    
    def generate(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        """
        生成聊天回复
        
        Args:
            system_prompt: 系统提示
            messages: 消息列表，格式为[{"role": "user", "content": "Hello"}, ...]
            
        Returns:
            生成的回复文本
        """
        if self.model is None or self.tokenizer is None:
            return "模型未正确加载"
            
        try:
            # 格式化消息
            formatted_messages = [{"role": "system", "content": system_prompt}]
            formatted_messages.extend(messages)
            
            # 生成回复
            response, _ = self.model.chat(
                self.tokenizer,
                formatted_messages,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                max_new_tokens=2048
            )
            
            return response
            
        except Exception as e:
            print(f"生成回复时出错: {e}")
            return "生成回复时出错，请重试"
    
    def generate_stream(self, system_prompt: str, messages: List[Dict[str, str]]):
        """
        流式生成聊天回复
        
        Args:
            system_prompt: 系统提示
            messages: 消息列表
            
        Yields:
            生成的文本片段
        """
        if self.model is None or self.tokenizer is None:
            yield "模型未正确加载"
            return
            
        try:
            # 格式化消息
            formatted_messages = [{"role": "system", "content": system_prompt}]
            formatted_messages.extend(messages)
            
            # 流式生成
            for response in self.model.chat_stream(
                self.tokenizer,
                formatted_messages,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                max_new_tokens=2048
            ):
                yield response
                
        except Exception as e:
            print(f"流式生成回复时出错: {e}")
            yield "生成回复时出错，请重试"
    
    def embeddings(self, text: str) -> Optional[torch.Tensor]:
        """
        获取文本的嵌入表示
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        if self.model is None or self.tokenizer is None:
            return None
            
        try:
            # 分词
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            # 获取嵌入
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                
            # 使用最后一层的隐藏状态作为嵌入
            last_hidden_state = outputs.hidden_states[-1]
            
            # 取平均值作为文本嵌入
            embeddings = last_hidden_state.mean(dim=1)
            
            return embeddings
            
        except Exception as e:
            print(f"获取嵌入向量时出错: {e}")
            return None 