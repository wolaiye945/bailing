import logging
import numpy as np
from pyrnnoise import RNNoise

logger = logging.getLogger(__name__)

class RNNoiseReducer:
    def __init__(self, config):
        self.enabled = config.get("enabled", False)
        self.sample_rate = config.get("sample_rate", 16000)
        self.control = config.get("control", 50) # 0-100, 50 is default
        
        if self.enabled:
            logger.info(f"Initializing RNNoise with sample_rate={self.sample_rate}, control={self.control}")
            try:
                self.denoiser = RNNoise(sample_rate=self.sample_rate)
            except Exception as e:
                logger.error(f"Failed to initialize RNNoise: {e}")
                self.enabled = False
        else:
            self.denoiser = None

    def process(self, data: bytes) -> bytes:
        if not self.enabled or not data:
            return data
        
        try:
            # Convert bytes to numpy array (Int16)
            audio_int16 = np.frombuffer(data, dtype=np.int16)
            # RNNoise expects [num_channels, num_samples]
            # Here we assume mono, so [1, num_samples]
            audio_data = audio_int16.reshape(1, -1)
            
            denoised_frames = []
            # denoise_chunk yields (speech_prob, denoised_frame)
            # denoised_frame is already Int16 if input was Int16? 
            # Let's check pyrnnoise behavior. It usually returns same type.
            for speech_prob, denoised_frame in self.denoiser.denoise_chunk(audio_data):
                denoised_frames.append(denoised_frame)
            
            if not denoised_frames:
                return data
            
            # Combine denoised frames
            # Each denoised_frame is [num_channels, frame_size]
            combined = np.concatenate(denoised_frames, axis=1)
            return combined.tobytes()
            
        except Exception as e:
            logger.error(f"Error in RNNoise processing: {e}")
            return data

def create_instance(config):
    return RNNoiseReducer(config)
