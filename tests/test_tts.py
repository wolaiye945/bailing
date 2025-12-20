import os
import sys
import logging

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from bailing import tts
from bailing.utils import read_config

logging.basicConfig(level=logging.DEBUG)

def test_kokoro_tts():
    config = read_config("config/config.yaml")
    # Manually adjust config for KOKOROTTS to match expected keys if necessary
    kokoro_config = config["TTS"]["KOKOROTTS"]
    # The class expects 'repo_id' and 'voice', but config has 'model_path' and 'voices'
    kokoro_params = {
        "repo_id": kokoro_config.get("model_path", "hexgrad/Kokoro-82M-v1.1-zh"),
        "voice": kokoro_config.get("voices", "zf_001"),
        "lang": kokoro_config.get("lang", "z"),
        "output_dir": "tmp"
    }
    
    print(f"Initializing KOKOROTTS with params: {kokoro_params}")
    try:
        kokoro = tts.KOKOROTTS(kokoro_params)
        text = "你好，我是百聆。很高兴见到你。"
        print(f"Generating TTS for: {text}")
        output_file = kokoro.to_tts(text)
        if output_file and os.path.exists(output_file):
            print(f"Successfully generated TTS: {output_file}")
        else:
            print("Failed to generate TTS output file.")
    except Exception as e:
        print(f"Error testing KOKOROTTS: {e}")

if __name__ == "__main__":
    # Set HF_ENDPOINT just in case
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    test_kokoro_tts()
