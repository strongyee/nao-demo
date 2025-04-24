#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import asyncio
import logging
import argparse
import concurrent.futures
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import websockets
from websockets.server import WebSocketServerProtocol

from audio_processing import AudioProcessor
from tts import OrpheusTTS
from llm import QwenChat
from vad import SileroVAD

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("server.log")
    ]
)
logger = logging.getLogger(__name__)

# 连接管理器
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocketServerProtocol] = {}
        self.audio_buffers: Dict[str, List[bytes]] = {}
        
    async def connect(self, websocket: WebSocketServerProtocol):
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        self.active_connections[client_id] = websocket
        self.audio_buffers[client_id] = []
        logger.info(f"客户端 {client_id} 已连接")
        return client_id
        
    async def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.audio_buffers:
            del self.audio_buffers[client_id]
        logger.info(f"客户端 {client_id} 已断开连接")
    
    async def send_text(self, client_id: str, message: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send(message)
            logger.debug(f"向客户端 {client_id} 发送消息: {message}")
    
    async def send_audio(self, client_id: str, audio_data: bytes):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send(audio_data)
            logger.debug(f"向客户端 {client_id} 发送音频数据: {len(audio_data)} 字节")
    
    def add_audio_chunk(self, client_id: str, chunk: bytes):
        if client_id in self.audio_buffers:
            self.audio_buffers[client_id].append(chunk)
    
    def get_audio_buffer(self, client_id: str) -> List[bytes]:
        if client_id in self.audio_buffers:
            buffer = self.audio_buffers[client_id]
            self.audio_buffers[client_id] = []
            return buffer
        return []

# 主服务器类
class TalkServer:
    def __init__(self, config_path: str = "config.json"):
        # 加载配置
        self.load_config(config_path)
        
        # 初始化管理器
        self.manager = ConnectionManager()
        
        # 初始化音频处理器
        self.audio_processor = AudioProcessor(
            sample_rate=self.config["audio"]["sample_rate"],
            channels=self.config["audio"]["channels"]
        )
        
        # 初始化VAD
        self.vad = SileroVAD(
            threshold=self.config["vad"]["threshold"],
            sampling_rate=self.config["audio"]["sample_rate"],
            device=self.config["device"]
        )
        
        # 初始化TTS
        self.tts = OrpheusTTS(
            model_path=self.config["tts"]["model_path"],
            device=self.config["device"]
        )
        
        # 如果Orpheus初始化失败，使用后备TTS
        if self.tts.engine is None:
            from tts import FallbackTTS
            logger.info("Orpheus TTS初始化失败，使用后备TTS引擎")
            self.tts = FallbackTTS()
        
        # 初始化LLM
        self.llm = QwenChat(
            model_name=self.config["llm"]["model_name"],
            device=self.config["device"]
        )
        
        # 线程池执行器（用于处理计算密集型任务）
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        
        # 用户会话状态
        self.client_states = {}
        
        # 初始化系统提示
        with open(self.config["llm"]["system_prompt_path"], "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

    def load_config(self, config_path: str):
        """加载配置文件"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
                
            # 如果未指定设备，自动选择
            if "device" not in self.config:
                self.config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
                
            logger.info(f"已加载配置，使用设备: {self.config['device']}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            # 使用默认配置
            self.config = {
                "server": {
                    "host": "0.0.0.0",
                    "port": 8080
                },
                "audio": {
                    "sample_rate": 16000,
                    "channels": 1
                },
                "vad": {
                    "threshold": 0.5
                },
                "tts": {
                    "model_path": "models/orpheus"
                },
                "llm": {
                    "model_name": "Qwen/Qwen2.5-7B-Chat",
                    "system_prompt_path": "prompts/system.txt"
                },
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            }
            
            # 写入默认配置
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)
    
    def initialize_client_state(self, client_id: str):
        """初始化客户端状态"""
        self.client_states[client_id] = {
            "conversation_history": [],
            "is_speaking": False,
            "last_utterance": "",
            "audio_buffer": [],
            "speaking_task": None
        }
    
    async def handle_audio(self, client_id: str, audio_chunks: List[bytes]):
        """处理接收到的音频数据"""
        if not audio_chunks:
            return
        
        # 将音频数据合并并转换为float数组
        audio_data = self.audio_processor.bytes_to_audio(b''.join(audio_chunks))
        
        # 使用VAD检测语音活动
        speech_frames = await asyncio.get_event_loop().run_in_executor(
            self.executor, 
            self.vad.process_audio, 
            audio_data
        )
        
        if speech_frames:
            # 检测到语音，添加到客户端的音频缓冲区
            client_state = self.client_states.get(client_id)
            if client_state:
                client_state["audio_buffer"].extend(speech_frames)
                
                # 检查是否需要打断当前的语音合成
                if client_state["is_speaking"] and len(speech_frames) > 5:
                    logger.info(f"检测到用户打断，停止当前回答")
                    if client_state["speaking_task"] and not client_state["speaking_task"].done():
                        client_state["speaking_task"].cancel()
                        client_state["is_speaking"] = False
                        
                # 检查是否有足够长的语音进行识别
                if len(client_state["audio_buffer"]) > 20:  # 约1秒的语音
                    # 运行语音识别
                    await self.process_speech(client_id)
    
    async def process_speech(self, client_id: str):
        """处理语音识别和生成回复"""
        client_state = self.client_states.get(client_id)
        if not client_state:
            return
        
        # 获取音频缓冲区
        audio_buffer = client_state["audio_buffer"]
        client_state["audio_buffer"] = []
        
        if not audio_buffer:
            return
        
        # 合并音频数据
        audio_data = np.concatenate(audio_buffer)
        
        # 执行语音识别
        text = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.audio_processor.recognize_speech,
            audio_data
        )
        
        if not text:
            return
        
        logger.info(f"客户端 {client_id} 的语音识别结果: {text}")
        
        # 向客户端发送识别结果
        await self.manager.send_text(client_id, json.dumps({
            "type": "asr_result",
            "text": text
        }))
        
        # 将用户输入添加到会话历史
        client_state["conversation_history"].append({"role": "user", "content": text})
        
        # 生成LLM回复
        response = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.llm.generate,
            self.system_prompt,
            client_state["conversation_history"]
        )
        
        if response:
            logger.info(f"LLM回复: {response}")
            
            # 添加到会话历史
            client_state["conversation_history"].append({"role": "assistant", "content": response})
            client_state["last_utterance"] = response
            
            # 向客户端发送文本回复
            await self.manager.send_text(client_id, json.dumps({
                "type": "llm_response",
                "text": response
            }))
            
            # 生成语音并发送
            client_state["is_speaking"] = True
            client_state["speaking_task"] = asyncio.create_task(
                self.generate_and_send_speech(client_id, response)
            )
    
    async def generate_and_send_speech(self, client_id: str, text: str):
        """生成语音并发送给客户端"""
        try:
            # 生成语音
            audio_data = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.tts.synthesize,
                text
            )
            
            if audio_data is None:
                logger.error("TTS合成失败")
                return
            
            # 将音频数据分成小块并发送
            chunk_size = 4096  # 每个块的大小
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i+chunk_size]
                await self.manager.send_audio(client_id, chunk)
                await asyncio.sleep(0.01)  # 给网络一些呼吸空间
            
            # 发送语音结束标记
            await self.manager.send_text(client_id, json.dumps({
                "type": "tts_complete"
            }))
            
        except asyncio.CancelledError:
            logger.info(f"语音合成任务被取消")
            # 发送语音被打断的标记
            await self.manager.send_text(client_id, json.dumps({
                "type": "tts_interrupted"
            }))
        except Exception as e:
            logger.error(f"生成或发送语音时出错: {e}")
        finally:
            # 更新客户端状态
            client_state = self.client_states.get(client_id)
            if client_state:
                client_state["is_speaking"] = False
    
    async def websocket_handler(self, websocket: WebSocketServerProtocol, path: str):
        """处理WebSocket连接"""
        client_id = await self.manager.connect(websocket)
        self.initialize_client_state(client_id)
        
        try:
            # 发送欢迎消息
            await self.manager.send_text(client_id, json.dumps({
                "type": "system",
                "text": "已连接到NAO交互服务器"
            }))
            
            # 处理消息
            async for message in websocket:
                # 检查是否为二进制数据(音频)或文本数据
                if isinstance(message, bytes):
                    # 处理音频数据
                    self.manager.add_audio_chunk(client_id, message)
                    await self.handle_audio(client_id, self.manager.get_audio_buffer(client_id))
                else:
                    # 处理文本消息
                    try:
                        data = json.loads(message)
                        logger.debug(f"收到客户端 {client_id} 的消息: {data}")
                        
                        if data.get("type") == "text":
                            # 直接处理文本输入
                            client_state = self.client_states.get(client_id)
                            if client_state:
                                client_state["conversation_history"].append({
                                    "role": "user", 
                                    "content": data.get("text", "")
                                })
                                
                                # 生成回复
                                response = await asyncio.get_event_loop().run_in_executor(
                                    self.executor,
                                    self.llm.generate,
                                    self.system_prompt,
                                    client_state["conversation_history"]
                                )
                                
                                if response:
                                    client_state["conversation_history"].append({
                                        "role": "assistant", 
                                        "content": response
                                    })
                                    
                                    # 发送回复
                                    await self.manager.send_text(client_id, json.dumps({
                                        "type": "llm_response",
                                        "text": response
                                    }))
                                    
                                    # 生成语音并发送
                                    client_state["is_speaking"] = True
                                    client_state["speaking_task"] = asyncio.create_task(
                                        self.generate_and_send_speech(client_id, response)
                                    )
                                    
                    except json.JSONDecodeError:
                        logger.warning(f"收到无效的JSON消息: {message}")
                    except Exception as e:
                        logger.error(f"处理消息时出错: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"客户端 {client_id} 连接已关闭")
        except Exception as e:
            logger.error(f"处理客户端 {client_id} 连接时发生错误: {e}")
        finally:
            await self.manager.disconnect(client_id)
            if client_id in self.client_states:
                del self.client_states[client_id]
    
    async def run(self):
        """启动WebSocket服务器"""
        host = self.config["server"]["host"]
        port = self.config["server"]["port"]
        
        server = await websockets.serve(
            self.websocket_handler,
            host,
            port
        )
        
        logger.info(f"服务器运行在 {host}:{port}")
        
        # 创建必要的目录
        os.makedirs("logs", exist_ok=True)
        
        # 无限运行直到关闭
        await server.wait_closed()


# 主函数
async def main():
    parser = argparse.ArgumentParser(description="NAO交互服务器")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    args = parser.parse_args()
    
    # 创建并运行服务器
    server = TalkServer(config_path=args.config)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main()) 