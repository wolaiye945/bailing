import asyncio
import logging
import os
import subprocess
import threading
import time
import uuid
from abc import ABC, ABCMeta, abstractmethod
from datetime import datetime
from gtts import gTTS
import edge_tts
import ChatTTS
import torch
import torchaudio
import soundfile as sf
from kokoro import KModel, KPipeline

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


class AbstractTTS(ABC):
    __metaclass__ = ABCMeta

    @abstractmethod
    def to_tts(self, text, username=None):
        pass

    def _generate_filename(self, output_file, extension=".wav", username=None):
        # 确保基础输出目录存在
        # output_file 可能是 config 中的 'tmp/'
        base_output = output_file if output_file else "tmp"
        
        # 处理可能的相对路径
        if not os.path.isabs(base_output):
            # 获取项目根目录（假设 tts.py 在 bailing 目录下）
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_output = os.path.join(project_root, base_output)
            
        if not os.path.exists(base_output):
            try:
                os.makedirs(base_output, exist_ok=True)
            except Exception as e:
                logger.error(f"创建基础目录失败: {base_output}, error: {e}")
        
        # 使用用户特定的子目录（如果提供了用户名）
        user_dir = base_output
        if username:
            user_dir = os.path.join(base_output, username)
            if not os.path.exists(user_dir):
                try:
                    os.makedirs(user_dir, exist_ok=True)
                    logger.info(f"创建用户目录: {user_dir}")
                except Exception as e:
                    logger.error(f"创建用户目录失败: {user_dir}, error: {e}")

        # 使用日期子目录
        date_str = time.strftime("%Y-%m-%d")
        date_dir = os.path.join(user_dir, date_str)
        if not os.path.exists(date_dir):
            try:
                os.makedirs(date_dir, exist_ok=True)
                logger.info(f"创建日期目录: {date_dir}")
            except Exception as e:
                logger.error(f"创建日期目录失败: {date_dir}, error: {e}")
            
        return os.path.join(date_dir, f"tts-{uuid.uuid4().hex}{extension}")


class GTTS(AbstractTTS):
    def __init__(self, config):
        self.output_file = config.get("output_file")
        self.lang = config.get("lang")

    def _log_execution_time(self, start_time):
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"执行时间: {execution_time:.2f} 秒")

    def to_tts(self, text, username=None):
        tmpfile = self._generate_filename(self.output_file, ".aiff", username)
        try:
            start_time = time.time()
            tts = gTTS(text=text, lang=self.lang)
            tts.save(tmpfile)
            self._log_execution_time(start_time)
            return tmpfile
        except Exception as e:
            logger.debug(f"生成TTS文件失败: {e}")
            return None


class MacTTS(AbstractTTS):
    """
    macOS 系统自带的TTS
    voice: say -v ? 可以打印所有语音
    """

    def __init__(self, config):
        super().__init__()
        self.voice = config.get("voice")
        self.output_file = config.get("output_file")

    def _log_execution_time(self, start_time):
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"执行时间: {execution_time:.2f} 秒")

    def to_tts(self, phrase, username=None):
        logger.debug(f"正在转换的tts：{phrase}")
        tmpfile = self._generate_filename(self.output_file, ".aiff", username)
        try:
            start_time = time.time()
            res = subprocess.run(
                ["say", "-v", self.voice, "-o", tmpfile, phrase],
                shell=False,
                universal_newlines=True,
            )
            self._log_execution_time(start_time)
            if res.returncode == 0:
                return tmpfile
            else:
                logger.info("TTS 生成失败")
                return None
        except Exception as e:
            logger.info(f"执行TTS失败: {e}")
            return None


class EdgeTTS(AbstractTTS):
    def __init__(self, config):
        self.output_file = config.get("output_file", "tmp/")
        self.voice = config.get("voice")

    def _log_execution_time(self, start_time):
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"Execution Time: {execution_time:.2f} seconds")

    async def text_to_speak(self, text, output_file):
        communicate = edge_tts.Communicate(text, voice=self.voice)  # Use your preferred voice
        await communicate.save(output_file)

    def to_tts(self, text, username=None):
        tmpfile = self._generate_filename(self.output_file, ".mp3", username)
        start_time = time.time()
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # 尝试使用 edge-tts 生成音频
                asyncio.run(self.text_to_speak(text, tmpfile))
                self._log_execution_time(start_time)
                return tmpfile
            except Exception as e:
                logger.warning(f"EdgeTTS attempt {attempt + 1} failed: {e}")
                if "403" in str(e):
                    # 如果是 403 错误，尝试等待一下再重试
                    time.sleep(1)
                    continue
                break
        
        # 如果 EdgeTTS 最终失败，尝试备选方案 GTTS
        logger.info("EdgeTTS failed, falling back to GTTS...")
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang='zh')
            tts.save(tmpfile)
            logger.info(f"Fallback to GTTS successful: {tmpfile}")
            return tmpfile
        except Exception as fallback_e:
            logger.error(f"Fallback to GTTS also failed: {fallback_e}")
            return None


class CHATTTS(AbstractTTS):
    def __init__(self, config):
        self.output_file = config.get("output_file", ".")
        self.chat = ChatTTS.Chat()
        self.chat.load(compile=False)  # Set to True for better performance
        self.rand_spk = self.chat.sample_random_speaker()

    def _log_execution_time(self, start_time):
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"Execution Time: {execution_time:.2f} seconds")

    def to_tts(self, text, username=None):
        tmpfile = self._generate_filename(self.output_file, ".wav", username)
        start_time = time.time()
        try:
            params_infer_code = ChatTTS.Chat.InferCodeParams(
                spk_emb=self.rand_spk,  # add sampled speaker
                temperature=.3,  # using custom temperature
                top_P=0.7,  # top P decode
                top_K=20,  # top K decode
            )
            params_refine_text = ChatTTS.Chat.RefineTextParams(
                prompt='[oral_2][laugh_0][break_6]',
            )
            wavs = self.chat.infer(
                [text],
                params_refine_text=params_refine_text,
                params_infer_code=params_infer_code,
            )
            # Ensure the audio is in int16 format and saved correctly
            audio_data = wavs[0]
            if audio_data.dtype != np.int16:
                if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                    audio_data = (audio_data * 32767).astype(np.int16)
            
            import soundfile as sf
            sf.write(tmpfile, audio_data, 24000, subtype='PCM_16')
            
            self._log_execution_time(start_time)
            return tmpfile
        except Exception as e:
            logger.error(f"Failed to generate TTS file: {e}")
            return None



# class KOKOROTTS(AbstractTTS):
#     def __init__(self, config):
#         from kokoro import KPipeline
#         self.output_file = config.get("output_file", ".")
#         self.lang = config.get("lang", "z")
#         self.pipeline = KPipeline(lang_code=self.lang)  # <= make sure lang_code matches voice
#         self.voice = config.get("voice", "zm_yunyang")
#
#     def _generate_filename(self, extension=".wav"):
#         return os.path.join(self.output_file, f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}")
#
#     def _log_execution_time(self, start_time):
#         end_time = time.time()
#         execution_time = end_time - start_time
#         logger.debug(f"Execution Time: {execution_time:.2f} seconds")
#
#     def to_tts(self, text):
#         tmpfile = self._generate_filename(".wav")
#         start_time = time.time()
#         try:
#             generator = self.pipeline(
#                 text, voice=self.voice,  # <= change voice here
#                 speed=1, split_pattern=r'\n+'
#             )
#             for i, (gs, ps, audio) in enumerate(generator):
#                 logger.debug(f"KOKOROTTS: i: {i}, gs：{gs}, ps：{ps}")  # i => index
#                 sf.write(tmpfile, audio, 24000)  # save each audio file
#             self._log_execution_time(start_time)
#             return tmpfile
#         except Exception as e:
#             logger.error(f"Failed to generate TTS file: {e}")
#             return None


class KOKOROTTS(AbstractTTS):
    _model_instance = None
    _instance_lock = threading.Lock()

    def __init__(self, config):
        """
        config keys:
          - repo_id: HuggingFace repo ID for Kokoro model (e.g. 'hexgrad/Kokoro-82M-v1.1-zh')
          - lang:      'z' for Chinese, 'a' for multilingual/IPA fallback
          - voice:     e.g. 'zf_001' or 'zm_010'
          - output_dir: directory to write wav files to
          - sample_rate: audio sample rate (default 24000)
          - zero_padding: number of zeros between segments (default 5000)
        """
        self.repo_id      = config.get("repo_id", "hexgrad/Kokoro-82M-v1.1-zh")
        self.output_dir   = config.get("output_dir") or config.get("output_file") or "tmp"
        self.lang         = config.get("lang", "z")
        self.voice        = config.get("voice", "zf_001")
        self.sample_rate  = config.get("sample_rate", 24000)

        # device selection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # load model if Chinese TTS
        with KOKOROTTS._instance_lock:
            if KOKOROTTS._model_instance is None:
                if self.lang == "z":
                    logger.info(f"Loading KOKOROTTS model from {self.repo_id}...")
                    # Update KModel.REPO_ID to the Chinese repo
                    KModel.REPO_ID = self.repo_id
                    try:
                        # Try to download the v1.1-zh model file
                        model_path = hf_hub_download(repo_id=self.repo_id, filename="kokoro-v1_1-zh.pth")
                        KOKOROTTS._model_instance = KModel(model=model_path).to(self.device).eval()
                    except Exception as e:
                        logger.warning(f"Failed to load kokoro-v1_1-zh.pth: {e}. Falling back to default.")
                        KOKOROTTS._model_instance = KModel().to(self.device).eval()
        self.model = KOKOROTTS._model_instance

        # set up pipelines
        self._setup_pipelines()

    def _setup_pipelines(self):
        # Main TTS pipeline
        self.pipeline = KPipeline(
            lang_code=self.lang,
            model=self.model,
            device=self.device
        )

    def _log_execution_time(self, start_time):
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"Execution Time: {execution_time:.2f} seconds")

    @staticmethod
    def _speed_callable(len_ps: int) -> float:
        """
        piecewise linear function to slow down generation for long inputs
        """
        speed = 0.8
        if len_ps <= 83:
            speed = 1.0
        elif len_ps < 183:
            speed = 1.0 - (len_ps - 83) / 500.0
        return speed * 1.1

    def to_tts(self, text, username=None):
        logger.debug(f"KOKOROTTS to_tts: {text}")
        try:
            output_file = self._generate_filename(self.output_dir, ".wav", username)
            generator = self.pipeline(
                text, voice=self.voice,
                speed=1, split_pattern=r'\n+'
            )
            all_audio = []
            for gs, ps, audio in generator:
                all_audio.append(audio.numpy())
            
            if all_audio:
                import numpy as np
                combined_audio = np.concatenate(all_audio)
                # Ensure the audio is in int16 format to avoid "Unknown WAVE format" (IEEE FLOAT) issues
                if combined_audio.dtype != np.int16:
                    # Scale to int16 range if it's float
                    if combined_audio.dtype == np.float32 or combined_audio.dtype == np.float64:
                        combined_audio = (combined_audio * 32767).astype(np.int16)
                sf.write(output_file, combined_audio, 24000, subtype='PCM_16')
                return output_file
            return None
        except Exception as e:
            logger.error(f"KOKOROTTS to_tts error: {e}")
            return None


def create_instance(class_name, *args, **kwargs):
    # 获取类对象
    cls = globals().get(class_name)
    if cls:
        # 创建并返回实例
        return cls(*args, **kwargs)
    else:
        raise ValueError(f"Class {class_name} not found")
