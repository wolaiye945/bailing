import sys
import os

def check_log(file_path, patterns):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # 尝试不同的编码
    encodings = ['utf-8', 'gbk', 'utf-16', 'ascii']
    content = None
    
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                lines = f.readlines()
                print(f"Successfully read with {enc}, total lines: {len(lines)}")
                
                for line in lines: # 检查所有行
                    for p in patterns:
                        if p in line:
                            print(f"FOUND: {line.strip()}")
                return
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error reading with {enc}: {e}")
            
    print("Failed to read log with any common encoding.")

if __name__ == "__main__":
    log_path = "tmp/bailing.log"
    search_patterns = [
        "尝试关联 WebRTC",
        "Robot 已关联 WebRTC 轨道",
        "WebRTC 已入队音频帧",
        "WebRTC.recv() 发送音频帧",
        "获取到 TTS 任务",
        "TTS 任务执行完成",
        "_tts_priority: 准备播放",
        "Player",
        "do_playing",
        "WebRTC 推送音频起始块",
        "WebRTC 已推送音频流",
        "WebRTC 轨道未设置，无法播放"
    ]
    check_log(log_path, search_patterns)
