import re

log_path = 'tmp/bailing.log'
patterns = [
    r'创建新 Robot 实例',
    r'正在关闭 Robot 实例',
    r'Robot 正在从',
    r'speak_and_play START',
    r'speak_and_play TTS DONE',
    r'获取到 TTS 任务',
    r'WebRTC 已推送音频流',
    r'WebRTC 已入队音频帧',
    r'WebRTC.recv',
    r'ASR 识别成功',
    r'识别到分句',
    r'VAD 状态更新',
    r'开始 ASR 识别',
    r'_tts_priority',
    r'Robot.__init__',
    r'Robot 已关联 WebRTC 轨道',
    r'WebRTC Offer 已存储',
    r'当前待关联 WebRTC 列表',
    r'WebSocket连接已建立',
    r'play file',
    r'WebRTC 轨道未设置'
]

with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if any(re.search(p, line) for p in patterns):
            print(line.strip())
