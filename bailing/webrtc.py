import asyncio
import logging
import queue
from aiortc import MediaStreamTrack
from av import AudioFrame
import numpy as np

logger = logging.getLogger(__name__)

class AudioTransformTrack(MediaStreamTrack):
    """
    WebRTC 音频轨道：
    1. 接收来自浏览器的音频 (recv) -> 放入 audio_queue 给 ASR。
    2. 接收来自 TTS 的 PCM (put_audio) -> 转换成 AudioFrame 发送给浏览器。
    """
    kind = "audio"

    def __init__(self, track=None):
        super().__init__()
        self.track = track  # 这是来自浏览器的输入轨道
        self.audio_queue = None  # ASR 消费的队列
        self.out_queue = asyncio.Queue()  # 发送给浏览器的队列
        self._running = True
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = None

    async def recv(self):
        """
        这个方法会被 WebRTC 引擎调用。
        如果本轨道被用作 pc.addTrack(track)，那么 recv() 返回的帧将发往浏览器。
        """
        if self.track:
            # 这是一个输入轨道包装器，我们需要读取输入并同时提供输出
            # 这种模式下，aiortc 可能需要我们主动循环读取 track.recv()
            # 但实际上，通常我们会分别为 input 和 output 创建轨道。
            pass

        # 作为输出轨道，从 out_queue 获取 TTS 帧发往浏览器
        frame = await self.out_queue.get()
        return frame

    async def start_input_loop(self):
        """
        独立循环：读取来自浏览器的音频并放入 ASR 队列
        """
        if not self.track:
            return
        
        logger.info("开始 WebRTC 输入音频循环")
        try:
            while self._running:
                frame = await self.track.recv()
                if self.audio_queue:
                    # 将 frame 转换为 16kHz, mono, s16 PCM
                    # aiortc 传过来的通常是 opus 解码后的，采样率可能不一致
                    # 我们需要重采样到 16000
                    # 简化处理：假设 ASR 能处理或者我们在这里转换
                    # 使用 av 的 resampler 或者简单的 ndarray 处理
                    
                    # 转换成 ndarray (frames, channels)
                    data = frame.to_ndarray()
                    # 如果是多声道转单声道
                    if data.ndim > 1 and data.shape[0] > 1:
                        data = np.mean(data, axis=0)
                    
                    # 如果采样率不是 16000，这里需要重采样
                    # 暂时假设是 16000 或者由 ASR 处理
                    pcm_bytes = data.astype(np.int16).tobytes()
                    self.audio_queue.put(pcm_bytes)
        except Exception as e:
            logger.error(f"WebRTC 输入循环异常: {e}")

    def put_audio(self, pcm_bytes):
        """
        TTS 调用：将 PCM 字节推送到输出队列
        """
        try:
            # 转换 PCM 字节为 AudioFrame
            # 假设 pcm_bytes 是 16kHz, 16bit, mono
            array = np.frombuffer(pcm_bytes, dtype=np.int16).reshape(1, -1)
            frame = AudioFrame.from_ndarray(array, format='s16', layout='mono')
            frame.sample_rate = 16000
            
            if self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(self.out_queue.put(frame), self.loop)
            else:
                # 这种情况下无法放入队列
                logger.warning("Event loop 未运行，音频帧丢弃")
        except Exception as e:
            logger.error(f"put_audio 异常: {e}")

    def stop(self):
        self._running = False
