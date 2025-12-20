from pydub import AudioSegment
import os

def test_pydub_wav():
    # Create a dummy wav file if possible, or just try to load an existing one
    # Actually, let's just check if pydub can find ffmpeg
    try:
        from pydub.utils import get_prober_name, get_player_name
        print(f"Prober: {get_prober_name()}")
        print(f"Player: {get_player_name()}")
    except Exception as e:
        print(f"Error checking pydub utils: {e}")

    # Try to load a wav file
    test_file = "test_audio.wav"
    # Create a simple wav file using wave module
    import wave
    import struct
    
    with wave.open(test_file, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(44100)
        for i in range(44100):
            value = int(32767.0 * 0.5)
            data = struct.pack('<h', value)
            f.writeframesraw(data)
            
    try:
        audio = AudioSegment.from_file(test_file, format="wav")
        print("Successfully loaded wav file with pydub")
        audio.export("test_output.wav", format="wav")
        print("Successfully exported wav file with pydub")
    except Exception as e:
        print(f"Failed to process wav with pydub: {e}")
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)
        if os.path.exists("test_output.wav"):
            os.remove("test_output.wav")

if __name__ == "__main__":
    test_pydub_wav()
