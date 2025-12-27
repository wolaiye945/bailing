import numpy as np
try:
    from pyrnnoise import RNNoise
    import traceback

    def test_rnnoise():
        try:
            # Test 16kHz
            print("Testing RNNoise with 16000Hz...")
            denoiser = RNNoise(sample_rate=16000)
            # Create 16000Hz dummy data (mono)
            # 480 samples at 48kHz is 10ms. 
            # 160 samples at 16kHz is 10ms.
            dummy_data = np.zeros((1, 1600), dtype=np.int16)
            for speech_prob, denoised_audio in denoiser.denoise_chunk(dummy_data):
                print(f"16kHz Speech probability: {speech_prob}")
                print(f"Denoised audio type: {denoised_audio.dtype}")
                break
            print("16kHz Test passed!")

        except Exception as e:
            print(f"16kHz Test failed: {e}")
            traceback.print_exc()

    if __name__ == "__main__":
        test_rnnoise()
except ImportError:
    print("pyrnnoise not installed")
