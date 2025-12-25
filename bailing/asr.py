import os
import uuid
import wave
from abc import ABC, abstractmethod
import logging
import threading
from datetime import datetime
import numpy as np

# Fix HF_ENDPOINT if it's set without protocol
if os.environ.get("HF_ENDPOINT") and not os.environ.get("HF_ENDPOINT").startswith("http"):
    os.environ["HF_ENDPOINT"] = "https://" + os.environ["HF_ENDPOINT"]

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


logger = logging.getLogger(__name__)


class ASR(ABC):
    @staticmethod
    def _save_audio_to_file(audio_data, file_path):
        """将音频数据保存为WAV文件"""
        try:
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b''.join(audio_data))
            logger.info(f"ASR识别文件录音保存到：{file_path}")
        except Exception as e:
            logger.error(f"保存音频文件时发生错误: {e}")
            raise

    @abstractmethod
    def recognizer(self, stream_in_audio, username=None):
        """处理输入音频流并返回识别的文本，子类必须实现"""
        pass


class FunASR(ASR):
    _model_instance = None
    _instance_lock = threading.Lock()

    def __init__(self, config):
        self.model_dir = config.get("model_dir")
        self.output_dir = config.get("output_file")

        with FunASR._instance_lock:
            if FunASR._model_instance is None:
                logger.info(f"Loading FunASR model from {self.model_dir}...")
                FunASR._model_instance = AutoModel(
                    model=self.model_dir,
                    vad_kwargs={"max_single_segment_time": 30000},
                    disable_update=True,
                    hub="hf"
                )
        self.model = FunASR._model_instance

    def recognizer(self, stream_in_audio, username=None):
        try:
            # 确保输出目录存在
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            # 使用用户特定的子目录（如果提供了用户名）
            base_dir = self.output_dir
            if username:
                base_dir = os.path.join(self.output_dir, username)
                if not os.path.exists(base_dir):
                    os.makedirs(base_dir)

            # 使用日期子目录
            date_str = datetime.now().strftime("%Y-%m-%d")
            date_dir = os.path.join(base_dir, date_str)
            if not os.path.exists(date_dir):
                os.makedirs(date_dir)

            tmpfile = os.path.join(date_dir, f"asr-{uuid.uuid4().hex}.wav")
            
            # 优化：在非调试模式下，我们可以并行保存文件和进行识别，或者直接使用内存数据
            # 这里的 SenseVoiceSmall 支持直接传入 numpy 数组
            audio_data = b''.join(stream_in_audio)
            if not audio_data:
                logger.warning("接收到空音频数据，跳过识别")
                return None, None
                
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # 异步保存文件（不阻塞识别）
            threading.Thread(target=self._save_audio_to_file, args=(stream_in_audio, tmpfile), daemon=True).start()

            res = self.model.generate(
                input=audio_np,
                cache={},
                language="auto",  # 语言选项: "zn", "en", "yue", "ja", "ko", "nospeech"
                use_itn=True,
                batch_size_s=60,
            )

            text = rich_transcription_postprocess(res[0]["text"])
            logger.info(f"识别文本: {text}")
            return text, tmpfile

        except Exception as e:
            logger.error(f"ASR识别过程中发生错误: {e}")
            return None, None


def create_instance(class_name, *args, **kwargs):
    # 获取类对象
    cls = globals().get(class_name)
    if cls:
        # 创建并返回实例
        return cls(*args, **kwargs)
    else:
        raise ValueError(f"Class {class_name} not found")