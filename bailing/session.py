import os
import logging
import uuid
from bailing.dialogue import Dialogue, Message
from bailing.memory import Memory

logger = logging.getLogger(__name__)

class Session:
    def __init__(self, user_info, memory_config, system_prompt):
        self.user_id = user_info.get("username", "default")
        self.user_info = user_info
        self.session_id = 0
        self.chat_lock = False
        
        # Initialize Memory and Dialogue for this session
        self.memory_config = memory_config.copy()
        self._setup_paths()
        
        if self.memory_config.get("enabled", True):
            self.memory = Memory(self.memory_config)
            self.memory_text = self.memory.get_memory()
        else:
            self.memory = None
            self.memory_text = ""
            
        self.prompt = system_prompt.replace("{memory}", self.memory_text).strip()
        self.dialogue = Dialogue(self.memory_config.get("dialogue_history_path", "tmp/dialogue"))
        self.dialogue.put(Message(role="system", content=self.prompt))
        
    def _setup_paths(self):
        if self.user_id != "default":
            user_path = self.user_id
            
            # Optimize dialogue history path
            orig_history_path = self.memory_config.get("dialogue_history_path", "tmp/dialogue")
            self.memory_config["dialogue_history_path"] = os.path.join(orig_history_path, user_path)
            
            # Optimize memory file path
            orig_memory_file = self.memory_config.get("memory_file", "tmp/memory.json")
            memory_dir = os.path.dirname(orig_memory_file)
            memory_base = os.path.basename(orig_memory_file)
            self.memory_config["memory_file"] = os.path.join(memory_dir, user_path, memory_base)
            
            # Ensure directories exist
            os.makedirs(os.path.dirname(self.memory_config["memory_file"]), exist_ok=True)
            os.makedirs(self.memory_config["dialogue_history_path"], exist_ok=True)

    def new_session(self):
        self.session_id += 1
        self.chat_lock = False
        return self.session_id

    def set_lock(self, locked: bool):
        self.chat_lock = locked

    def is_locked(self):
        return self.chat_lock

    def get_dialogue_history(self, max_history=15):
        return self.dialogue.get_llm_dialogue(max_history=max_history)

    def add_message(self, role, content):
        self.dialogue.put(Message(role=role, content=content))
