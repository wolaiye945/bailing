import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from funasr import AutoModel
import logging

logging.basicConfig(level=logging.INFO)

try:
    print("Attempting to load SenseVoiceSmall...")
    model = AutoModel(
        model="FunAudioLLM/SenseVoiceSmall",
        hub="hf"
    )
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
