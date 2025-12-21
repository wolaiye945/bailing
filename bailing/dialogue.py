import os.path
import uuid
from typing import List, Dict
from datetime import datetime
from bailing.utils import write_json_file


class Message:
    def __init__(self, role: str, content: str = None, uniq_id: str = None, start_time: datetime = None, end_time: datetime = None,
                 audio_file: str = None, tts_file: str = None, vad_status: list = None, tool_calls = None, tool_call_id=None):
        self.uniq_id = uniq_id if uniq_id is not None else str(uuid.uuid4())
        self.role = role
        self.content = content
        self.start_time = start_time
        self.end_time = end_time
        self.audio_file = audio_file
        self.tts_file = tts_file
        self.vad_status = vad_status
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id


class Dialogue:
    def __init__(self, dialogue_history_path):
        self.dialogue_history_path = dialogue_history_path
        self.dialogue: List[Message] = []
        # 获取当前时间
        self.current_time  = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    def put(self, message: Message):
        self.dialogue.append(message)

    def get_llm_dialogue(self, max_history: int = None) -> List[Dict[str, str]]:
        dialogue = []
        for m in self.dialogue:
            if m.tool_calls is not None:
                dialogue.append({"role": m.role, "tool_calls": m.tool_calls})
            elif m.role == "tool":
                dialogue.append({"role": m.role, "tool_call_id": m.tool_call_id, "content": m.content})
            else:
                dialogue.append({"role": m.role, "content": m.content})
        
        if max_history and len(dialogue) > max_history:
            # 保持系统提示词（通常是第一个消息）
            system_message = None
            if dialogue and dialogue[0]["role"] == "system":
                system_message = dialogue[0]
                remaining_dialogue = dialogue[1:]
                # 取最后的 max_history - 1 条消息
                truncated = remaining_dialogue[-(max_history - 1):]
                return [system_message] + truncated
            else:
                return dialogue[-max_history:]
                
        return dialogue

    def dump_dialogue(self):
        dialogue = []
        for d in self.get_llm_dialogue():
            if d["role"] not in ("user", "assistant"):
                continue
            dialogue.append(d)
        
        # 使用日期子目录
        date_str = datetime.now().strftime("%Y-%m-%d")
        date_dir = os.path.join(self.dialogue_history_path, date_str)
        if not os.path.exists(date_dir):
            os.makedirs(date_dir)
            
        file_name = os.path.join(date_dir, f"dialogue-{self.current_time}.json")
        write_json_file(file_name, dialogue)

if __name__ == "__main__":
    d = Dialogue("../tmp/")
    d.put(Message(role="user", content="你好"))
    d.dump_dialogue()