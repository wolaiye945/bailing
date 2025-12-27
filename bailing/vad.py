import os
import uuid
import wave
from abc import ABC, abstractmethod
import logging
from datetime import datetime

import numpy as np
import torch
from silero_vad import load_silero_vad, VADIterator
from funasr import AutoModel

logger = logging.getLogger(__name__)


class VAD(ABC):
    def __init__(self, config):
        self.original_threshold = config.get("threshold", 0.5)

    @abstractmethod
    def is_vad(self, data):
        pass

    def set_threshold(self, threshold):
        pass

    def reset_states(self):
        pass


class FunASRVAD(VAD):
    def __init__(self, config):
        super().__init__(config)
        self.model_dir = config.get("model_dir", "fsmn-vad")
        self.sampling_rate = config.get("sampling_rate", 16000)
        self.threshold = config.get("threshold", 0.5)
        self.max_single_segment_time = config.get("max_single_segment_time", 30000)
        self.min_speech_duration_ms = config.get("min_speech_duration_ms", 300) # 默认 300ms 以下认为是噪声
        
        logger.info(f"Loading FunASR VAD model from {self.model_dir}...")
        self.model = AutoModel(
            model=self.model_dir,
            disable_update=True,
            hub="ms" # Use modelscope for iic models
        )
        self.cache = {}
        self.is_speaking = False
        self.speech_start_time = 0

    def is_vad(self, data):
        """
        FunASR VAD returns list of segments [[start, end], ...]
        For streaming, we need to track state.
        """
        try:
            audio_int16 = np.frombuffer(data, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            # FunASR VAD expects 16k sampling rate
            res = self.model.generate(
                input=audio_float32,
                cache=self.cache,
                chunk_size=len(audio_float32) * 1000 // self.sampling_rate, # ms
                sampling_rate=self.sampling_rate,
                threshold=self.threshold # 动态传入阈值
            )
            
            if not res or not res[0]["value"]:
                return None

            segments = res[0]["value"]
            
            # 记录当前chunk的时间戳（近似）
            # 注意：FunASR 返回的是毫秒
            
            if not self.is_speaking and len(segments) > 0:
                # 发现可能的语音开始
                for seg in segments:
                    if seg[0] != -1:
                        self.is_speaking = True
                        self.speech_start_time = seg[0]
                        return {"start": seg[0]}
            
            if self.is_speaking:
                for seg in segments:
                    if seg[1] != -1: # 语音段结束
                        duration = seg[1] - self.speech_start_time
                        self.is_speaking = False
                        
                        if duration < self.min_speech_duration_ms:
                            logger.info(f"检测到语音结束，但时长太短 ({duration}ms < {self.min_speech_duration_ms}ms)，判定为噪声并丢弃")
                            return {"cancel": True, "end": seg[1]} # 返回 cancel 标志
                        
                        return {"end": seg[1]}
            
            return None

        except Exception as e:
            logger.error(f"Error in FunASR VAD processing: {e}")
            return None

    def reset_states(self):
        self.cache = {}
        self.is_speaking = False
        logger.debug("FunASR VAD states reset.")

    def set_threshold(self, threshold):
        """
        Update the VAD threshold. For FunASR, we use it to gate output.
        """
        self.threshold = threshold
        logger.debug(f"FunASR VAD threshold updated to: {threshold}")

class SileroVAD(VAD):
    def __init__(self, config):
        print("SileroVAD", config)
        self.model = load_silero_vad()
        self.sampling_rate = config.get("sampling_rate")
        self.threshold = config.get("threshold")
        self.original_threshold = self.threshold
        self.min_silence_duration_ms = config.get("min_silence_duration_ms")
        self.vad_iterator = VADIterator(self.model,
                            threshold=self.threshold,
                            sampling_rate=self.sampling_rate,
                            min_silence_duration_ms=self.min_silence_duration_ms)
        logger.debug(f"VAD Iterator initialized with model {self.model}")

    def set_threshold(self, threshold):
        """
        Dynamically update the VAD threshold.
        """
        try:
            self.threshold = threshold
            self.vad_iterator.threshold = threshold
            logger.debug(f"VAD threshold updated to: {threshold}")
        except Exception as e:
            logger.error(f"Error updating VAD threshold: {e}")

    @staticmethod
    def int2float(sound):
        """
        Convert int16 audio data to float32.
        """
        sound = sound.astype(np.float32) / 32768.0
        return sound

    def is_vad(self, data):
        try:
            audio_int16 = np.frombuffer(data, dtype=np.int16)
            audio_float32 = self.int2float(audio_int16)
            vad_output = self.vad_iterator(torch.from_numpy(audio_float32))
            if vad_output is not None:
                logger.debug(f"VAD output: {vad_output}")
            return vad_output
        except Exception as e:
            logger.error(f"Error in VAD processing: {e}")
            return None

    def reset_states(self):
        try:
            self.vad_iterator.reset_states()  # Reset model states after each audio
            logger.debug("VAD states reset.")
        except Exception as e:
            logger.error(f"Error resetting VAD states: {e}")


def create_instance(class_name, *args, **kwargs):
    # 获取类对象
    cls = globals().get(class_name)
    if cls:
        # 创建并返回实例
        return cls(*args, **kwargs)
    else:
        raise ValueError(f"Class {class_name} not found")
