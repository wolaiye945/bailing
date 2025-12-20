import os
import logging
from pydub import AudioSegment

logging.basicConfig(level=logging.DEBUG)

def test_pydub():
    # Find a wav file in tmp/
    tmp_dir = "tmp"
    wav_files = [f for f in os.listdir(tmp_dir) if f.endswith(".wav")]
    if not wav_files:
        print("No wav files found in tmp/")
        return
    
    test_file = os.path.join(tmp_dir, wav_files[0])
    print(f"Testing with file: {test_file}")
    
    try:
        audio = AudioSegment.from_file(test_file)
        print("Successfully loaded audio with pydub")
        output_file = test_file + ".test.wav"
        audio.export(output_file, format="wav")
        print(f"Successfully exported to {output_file}")
        if os.path.exists(output_file):
            os.remove(output_file)
    except Exception as e:
        print(f"Pydub failed: {e}")

if __name__ == "__main__":
    test_pydub()
