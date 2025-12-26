import asyncio
import edge_tts
import os

async def test_edge_tts():
    text = "你好，测试一下语音合成。"
    voice = "zh-CN-XiaoxiaoNeural"
    output_file = "test_edge_tts.mp3"
    
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)
        print(f"Success! Saved to {output_file}")
        if os.path.exists(output_file):
            os.remove(output_file)
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_edge_tts())
