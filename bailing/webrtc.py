import asyncio
import logging
import queue
from fractions import Fraction
from aiortc import MediaStreamTrack
from av import AudioFrame, AudioResampler
import numpy as np

logger = logging.getLogger(__name__)

class AudioTransformTrack(MediaStreamTrack):
    """
    WebRTC 音频轨道：
    1. 接收来自浏览器的音频 (recv) -> 放入 audio_queue 给 ASR。
    2. 接收来自 TTS 的 PCM (put_audio) -> 转换成 AudioFrame 发送给浏览器。
    """
    kind = "audio"

    def __init__(self, track=None, loop=None):
        super().__init__()
        self.track = track  # 这是来自浏览器的输入轨道
        self.audio_queue = None  # ASR 消费的队列
        self.out_queue = None  # 延迟初始化
        self._running = True
        self.resampler = AudioResampler(format='s16', layout='mono', rate=16000)
        self._buffer = bytearray()  # 缓冲区，用于确保输出给 VAD 的块足够大
        self._min_chunk_size = 512 * 2  # 16kHz, 16bit, mono 下，512 samples = 1024 bytes
        
        # 为输出轨道增加 PTS 管理
        self._next_pts = 0
        self._sample_rate = 16000
        
        self.loop = loop
        if not self.loop:
            try:
                self.loop = asyncio.get_event_loop()
            except RuntimeError:
                self.loop = None
        
        if self.loop:
            # 在 loop 所在的线程中创建 Queue
            if self.loop.is_running():
                def create_queue():
                    self.out_queue = asyncio.Queue()
                self.loop.call_soon_threadsafe(create_queue)
            else:
                self.out_queue = asyncio.Queue()
        
        logger.info(f"AudioTransformTrack 初始化: loop={self.loop}, out_queue_created={self.out_queue is not None}")

    async def recv(self):
        """
        如果本轨道被用作 pc.addTrack(track)，那么 recv() 返回的帧将发往浏览器。
        """
        if not self.out_queue:
            logger.debug("WebRTC.recv() 等待队列初始化...")
            # 等待队列初始化
            for _ in range(10):
                if self.out_queue:
                    break
                await asyncio.sleep(0.1)
            else:
                logger.error("WebRTC.recv() 等待队列初始化超时")
                # 返回静音帧或抛出异常
                return await super().recv()
        
        # 作为输出轨道，从 out_queue 获取 TTS 帧发往浏览器
        try:
            # 这里的 recv() 需要控制频率，否则会一次性把队列里的所有帧都发出去
            # 每帧 20ms (320 samples @ 16kHz)
            frame = await self.out_queue.get()
            
            # 增加一个标记，确认是从 out_queue 拿到的
            # logger.debug(f"WebRTC.recv() 从队列获取到帧: pts={frame.pts}, samples={frame.samples}, queue_size={self.out_queue.qsize()}")
            
            # 严格控制频率：记录发送时间
            now = asyncio.get_event_loop().time()
            if hasattr(self, "_last_send_time"):
                elapsed = now - self._last_send_time
                wait = 0.020 - elapsed
                if wait > 0:
                    await asyncio.sleep(wait)
            
            self._last_send_time = asyncio.get_event_loop().time()
            return frame
        except Exception as e:
            logger.error(f"WebRTC.recv() 异常: {e}")
            return await super().recv()

    async def start_input_loop(self):
        """
        开始从 remoteTrack 获取音频帧
        """
        logger.info("AudioTransformTrack.start_input_loop() 启动")
        if not self.out_queue:
            self.out_queue = asyncio.Queue()
        
        logger.info("等待 WebRTC 输入轨道...")
        # 等待最多 10 秒
        for _ in range(100):
            if self.track:
                break
            await asyncio.sleep(0.1)

        if not self.track:
            logger.error("WebRTC 输入轨道未在超时内设置，启动输入循环失败")
            return
        
        logger.info("开始 WebRTC 输入音频循环")
        recv_count = 0
        try:
            while self._running:
                # 从浏览器接收音频帧
                if recv_count == 0:
                    logger.info("正在等待第一帧 WebRTC 音频数据 (recv)...")
                
                frame = await self.track.recv()
                recv_count += 1
                
                if recv_count == 1:
                    logger.info("WebRTC 收到第一帧音频数据")
                if recv_count % 100 == 0:
                    logger.debug(f"WebRTC 已接收 {recv_count} 帧音频")

                if self.audio_queue:
                    # 使用 PyAV Resampler 进行重采样和格式转换
                    # 确保输出为 16kHz, mono, s16
                    resampled_frames = self.resampler.resample(frame)
                    for resampled_frame in resampled_frames:
                        data = resampled_frame.to_ndarray()
                        pcm_bytes = data.astype(np.int16).tobytes()
                        
                        # 放入缓冲区
                        self._buffer.extend(pcm_bytes)
                        
                        # 当缓冲区超过最小块大小时，取出并放入队列
                        while len(self._buffer) >= self._min_chunk_size:
                            chunk = bytes(self._buffer[:self._min_chunk_size])
                            self._buffer = self._buffer[self._min_chunk_size:]
                            
                            # 计算音量（RMS）
                            audio_data = np.frombuffer(chunk, dtype=np.int16)
                            rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
                            
                            if recv_count % 100 == 0:
                                 logger.debug(f"WebRTC 接收中: count={recv_count}, RMS={rms:.2f}, queue={id(self.audio_queue)}")
                            
                            self.audio_queue.put(chunk)
                else:
                    if recv_count % 100 == 0:
                        logger.warning(f"WebRTC 收到数据但 audio_queue 未就绪: count={recv_count}")
        except Exception as e:
            logger.error(f"WebRTC 输入循环异常: {e}", exc_info=True)

    def put_audio(self, pcm_bytes):
        """
        接收来自 TTS 的 PCM 数据并放入 out_queue
        """
        logger.debug(f"WebRTC.put_audio() 收到 PCM 数据: {len(pcm_bytes)} bytes")
        if not self.out_queue:
            logger.warning("WebRTC.out_queue 未初始化，丢弃音频数据")
            return
            
        try:
            # 转换为 AudioFrame
            # 假设 pcm_bytes 是 16kHz, 16bit, mono
            audio_data = np.frombuffer(pcm_bytes, dtype=np.int16)
            
            # 将数据切分为 20ms 的帧 (320 samples at 16kHz)
            chunk_samples = 320
            chunks_sent = 0
            for i in range(0, len(audio_data), chunk_samples):
                chunk = audio_data[i:i + chunk_samples]
                # 如果最后一帧不够长，补 0
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
                
                array = chunk.reshape(1, -1)
                frame = AudioFrame.from_ndarray(array, format='s16', layout='mono')
                frame.sample_rate = self._sample_rate
                frame.pts = self._next_pts
                frame.time_base = Fraction(1, self._sample_rate)
                self._next_pts += chunk_samples
                
                if self.loop and self.loop.is_running():
                    # 计算当前块的 RMS，确认不是静音
                    rms = np.sqrt(np.mean(chunk.astype(np.float32)**2))
                    if i == 0:
                        logger.debug(f"WebRTC 推送音频起始块: RMS={rms:.2f}")
                    
                    asyncio.run_coroutine_threadsafe(self.out_queue.put(frame), self.loop)
                    chunks_sent += 1
                else:
                    logger.warning(f"Event loop 未运行或已停止，音频帧丢弃: loop={self.loop}")
                    break
            
            logger.info(f"WebRTC 已入队音频帧: samples={len(audio_data)}, chunks={chunks_sent}, total_pts={self._next_pts}")
        except Exception as e:
            logger.error(f"put_audio 异常: {e}", exc_info=True)

    def stop(self):
        self._running = False
