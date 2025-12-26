import logging
import platform
import queue
import subprocess
import threading
import wave
import pyaudio
import json
import os
from pydub import AudioSegment
import pygame
import sounddevice as sd
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio

logger = logging.getLogger(__name__)


class AbstractPlayer(object):
    def __init__(self, *args, **kwargs):
        super(AbstractPlayer, self).__init__()
        self.is_playing = False
        self.play_queue = queue.Queue()
        self._stop_event = threading.Event()
        self.consumer_thread = threading.Thread(target=self._playing)
        self.consumer_thread.start()

    @staticmethod
    def to_wav(audio_file):
        if audio_file.endswith(".wav"):
            return audio_file
        tmp_file = audio_file + ".wav"
        try:
            # 记录转换开始
            logger.debug(f"正在将 {audio_file} 转换为 wav 格式...")
            
            # 检查文件是否存在
            if not os.path.exists(audio_file):
                logger.error(f"转换失败：源文件不存在 {audio_file}")
                return audio_file
                
            audio = AudioSegment.from_file(audio_file)
            audio.export(tmp_file, format="wav")
            
            # 验证转换后的文件
            if os.path.exists(tmp_file) and os.path.getsize(tmp_file) > 0:
                logger.debug(f"转换成功: {tmp_file} ({os.path.getsize(tmp_file)} bytes)")
                return tmp_file
            else:
                logger.error(f"转换后的文件无效或为空: {tmp_file}")
                return audio_file
        except Exception as e:
            logger.error(f"转换音频到 wav 失败: {e}", exc_info=True)
            return audio_file

    def _playing(self):
        while not self._stop_event.is_set():
            try:
                data = self.play_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            self.is_playing = True
            try:
                self.do_playing(data)
            except Exception as e:
                logger.error(f"播放音频失败: {e}")
            finally:
                self.play_queue.task_done()
                self.is_playing = False

    def play(self, data):
        logger.info(f"play file {data}")
        audio_file = self.to_wav(data)
        self.play_queue.put(audio_file)

    def stop(self):
        self._clear_queue()

    def shutdown(self):
        self._clear_queue()
        self._stop_event.set()
        if self.consumer_thread.is_alive():
            self.consumer_thread.join()

    def get_playing_status(self):
        """正在播放和队列非空，为正在播放状态"""
        return self.is_playing or (not self.play_queue.empty())

    def _clear_queue(self):
        with self.play_queue.mutex:
            self.play_queue.queue.clear()

    def do_playing(self, audio_file):
        """播放音频的具体实现，由子类实现"""
        raise NotImplementedError("Subclasses must implement do_playing")


class CmdPlayer(AbstractPlayer):
    def __init__(self, *args, **kwargs):
        super(CmdPlayer, self).__init__(*args, **kwargs)
        self.p = pyaudio.PyAudio()

    def do_playing(self, audio_file):
        system = platform.system()
        cmd = ["afplay", audio_file] if system == "Darwin" else ["play", audio_file]
        logger.debug(f"Executing command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, shell=False, universal_newlines=True)
            logger.debug(f"播放完成：{audio_file}")
        except subprocess.CalledProcessError as e:
            logger.error(f"命令执行失败: {e}")
        except Exception as e:
            logger.error(f"未知错误: {e}")


class PyaudioPlayer(AbstractPlayer):
    def __init__(self, *args, **kwargs):
        super(PyaudioPlayer, self).__init__(*args, **kwargs)
        self.p = pyaudio.PyAudio()

    def do_playing(self, audio_file):
        chunk = 1024
        try:
            with wave.open(audio_file, 'rb') as wf:
                stream = self.p.open(format=self.p.get_format_from_width(wf.getsampwidth()),
                                     channels=wf.getnchannels(),
                                     rate=wf.getframerate(),
                                     output=True)
                data = wf.readframes(chunk)
                while data:
                    stream.write(data)
                    data = wf.readframes(chunk)
                stream.stop_stream()
                stream.close()
            logger.debug(f"播放完成：{audio_file}")
        except Exception as e:
            logger.error(f"播放音频失败: {e}")

    def stop(self):
        super().stop()
        if self.p:
            self.p.terminate()


class PygamePlayer(AbstractPlayer):
    def __init__(self, *args, **kwargs):
        super(PygamePlayer, self).__init__(*args, **kwargs)
        pygame.mixer.init()

    def do_playing(self, audio_file):
        try:
            # Use pydub to load the audio file as it's more robust than the standard wave module
            audio = AudioSegment.from_file(audio_file)
            
            # Export to a temporary buffer or use sounddevice/pyaudio to play
            # For simplicity and compatibility, we'll use pydub's data
            samples = np.array(audio.get_array_of_samples())
            
            # Ensure we have the right shape and type for sounddevice
            if audio.channels > 1:
                samples = samples.reshape((-1, audio.channels))
            
            sd.play(samples, samplerate=audio.frame_rate)
            sd.wait()
            logger.debug(f"播放完成：{audio_file}")
        except Exception as e:
            logger.error(f"播放音频失败: {e}")

    def get_playing_status(self):
        """正在播放和队列非空，为正在播放状态"""
        return self.is_playing or (not self.play_queue.empty()) or pygame.mixer.music.get_busy()

    def stop(self):
        super().stop()
        pygame.mixer.music.stop()

class PygameSoundPlayer(AbstractPlayer):
    """支持预加载"""
    def __init__(self, *args, **kwargs):
        super(PygameSoundPlayer, self).__init__(*args, **kwargs)
        pygame.mixer.init()

    def do_playing(self, current_sound):
        try:
            logger.debug("PygameSoundPlayer 播放音频中")
            current_sound.play()  # 播放音频
            while pygame.mixer.get_busy(): #current_sound.get_busy():  # 检查当前音频是否正在播放
                pygame.time.Clock().tick(100)  # 每秒检查100次
            del current_sound
            logger.debug(f"PygameSoundPlayer 播放完成")
        except Exception as e:
            logger.error(f"播放音频失败: {e}")

    def play(self, data):
        logger.info(f"play file {data}")
        audio_file = self.to_wav(data)
        sound = pygame.mixer.Sound(audio_file)
        self.play_queue.put(sound)

    def stop(self):
        super().stop()


class SoundDevicePlayer(AbstractPlayer):
    def do_playing(self, audio_file):
        try:
            wf = wave.open(audio_file, 'rb')
            data = wf.readframes(wf.getnframes())
            sd.play(np.frombuffer(data, dtype=np.int16), samplerate=wf.getframerate())
            sd.wait()
            logger.debug(f"播放完成：{audio_file}")
        except Exception as e:
            logger.error(f"播放音频失败: {e}")

    def stop(self):
        super().stop()
        sd.stop()

class WebSocketPlayer(AbstractPlayer):
    """通过WebSocket发送音频到前端"""

    def __init__(self, *args, **kwargs):
        super(WebSocketPlayer, self).__init__(*args, **kwargs)

        self.websocket = None
        self.loop = None
        self.playing_status = False
        self.lock = threading.Lock()  # 添加线程锁
        self._playback_finished_event = threading.Event()
        self._playback_finished_event.set()

    def init(self, websocket: WebSocket, loop):
        self.websocket = websocket
        self.loop = loop

    def get_playing_status(self):
        """正在播放和队列非空，为正在播放状态"""
        return self.playing_status

    def set_playing_status(self, status):
        """正在播放和队列非空，为正在播放状态"""
        self.playing_status = status
        if not status:
            logger.debug("WebSocketPlayer: playback finished signal received")
            self._playback_finished_event.set()
        else:
            logger.debug("WebSocketPlayer: playback started signal received")
            self._playback_finished_event.clear()

    def do_playing(self, audio_file):
        try:
            if not self.websocket or self.websocket.client_state.value != 1:
                logger.warning(f"WebSocket 未连接，跳过播放: {audio_file}")
                return

            with open(audio_file, "rb") as f:
                wav_data = f.read()
            
            if not wav_data:
                logger.warning(f"音频文件为空: {audio_file}")
                return

            logger.info(f"WebSocket 准备发送音频文件：{audio_file} ({len(wav_data)} bytes)")
            
            # 发送调试状态，确保前端知道后端已经开始动作
            asyncio.run_coroutine_threadsafe(
                self.websocket.send_text(json.dumps({
                    "type": "debug", 
                    "message": f"后端正在推送音频二进制流: {os.path.basename(audio_file)} ({len(wav_data)} 字节)"
                })),
                self.loop
            )
            
            self._playback_finished_event.clear()

            # 确保使用 bytes 类型发送
            asyncio.run_coroutine_threadsafe(
                self.websocket.send_bytes(wav_data),
                self.loop
            )

            # 等待前端返回播放完成信号
            finished = self._playback_finished_event.wait(timeout=60)
            if not finished:
                logger.warning(f"等待前端播放音频超时 (60s): {audio_file}")
            else:
                logger.debug(f"前端播放音频确认完成: {audio_file}")

        except Exception as e:
            logger.error(f"WebSocketPlayer 播放任务异常: {e}", exc_info=True)
            self._playback_finished_event.set() 

    def interrupt(self):
        """异步发送中断命令"""
        try:
            if self.websocket and self.websocket.client_state.value == 1:  # 1 = CONNECTED
                asyncio.run_coroutine_threadsafe(
                    self.websocket.send_text(json.dumps({"type": "interrupt"})),
                    self.loop
                )
            else:
                logger.warning("尝试中断时 WebSocket 未连接")
        except Exception as e:
            logger.error(f"发送中断命令失败: {e}")

    def send_messages(self, messages):
        logger.info(f"send_messages: {messages}")
        if not self.websocket or self.websocket.client_state.value != 1:  # 1 = CONNECTED
            logger.warning("发送消息时 WebSocket 未连接")
            return

        data = {
            "type": "update_dialogue",
            "data": messages if isinstance(messages, list) else [messages]
        }
        try:
            asyncio.run_coroutine_threadsafe(
                self.websocket.send_text(json.dumps(data)),
                self.loop
            )
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

    def send_status(self, status):
        """发送状态更新"""
        if not self.websocket or self.websocket.client_state.value != 1:
            return
        
        data = {
            "type": "status_update",
            "status": status
        }
        try:
            asyncio.run_coroutine_threadsafe(
                self.websocket.send_text(json.dumps(data)),
                self.loop
            )
        except Exception as e:
            logger.error(f"发送状态更新失败: {e}")

    def stop(self):
        """停止播放器"""
        try:
            if self.websocket and self.websocket.client_state.value == 1:  # 1 = CONNECTED
                asyncio.run_coroutine_threadsafe(
                    self.websocket.send_text(json.dumps({"type": "interrupt"})),
                    self.loop
                )
            #     await self.websocket.send_text(json.dumps({"type": "interrupt"}))
            #     # 关闭连接
            #     await self.websocket.close()
            # self.websocket = None
        except Exception as e:
            logger.error(f"停止播放器失败: {e}")


def create_instance(class_name, *args, **kwargs):
    # 获取类对象
    cls = globals().get(class_name)
    if cls:
        # 创建并返回实例
        print(args,kwargs)
        return cls(*args, **kwargs)
    else:
        raise ValueError(f"Class {class_name} not found")