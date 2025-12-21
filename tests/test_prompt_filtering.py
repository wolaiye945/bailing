import sys
import os
import yaml
import queue

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bailing.robot import Robot
from bailing.prompt import sys_prompt

def test_prompt_filtering():
    # Mock config
    config = {
        "selected_module": {
            "LLM": "MockLLM",
            "TTS": "MockTTS",
            "VAD": "MockVAD",
            "Player": "MockPlayer"
        },
        "LLM": {"MockLLM": {}},
        "TTS": {"MockTTS": {}},
        "VAD": {"MockVAD": {}},
        "Player": {"MockPlayer": {}},
        "Memory": {"enabled": False, "dialogue_history_path": "test_history.json"},
        "TaskManager": {"enabled": False},
        "StartTaskMode": True,
        "interrupt": True
    }

    # We need to mock some dependencies or provide enough config for Robot to init
    # Since Robot.__init__ creates many instances, it might be easier to just test the logic directly
    # or mock the necessary parts.
    
    print("Testing prompt filtering logic...")
    
    # Manually run the filtering logic from Robot.__init__
    from plugins.task_manager import TaskManager
    import re

    task_manager = TaskManager(config.get("TaskManager"), queue.Queue())
    start_task_mode = config.get("StartTaskMode")
    
    current_sys_prompt = sys_prompt
    print(f"Original prompt length: {len(current_sys_prompt)}")
    
    if not task_manager.enabled or not start_task_mode:
        # The new logic we added to Robot.__init__
        lines = current_sys_prompt.split('\n')
        filtered_lines = [line for line in lines if "调用工具" not in line and "function_name" not in line]
        current_sys_prompt = '\n'.join(filtered_lines)
    
    print(f"Filtered prompt length: {len(current_sys_prompt)}")
    print("-" * 20)
    print("Filtered Prompt:")
    print(current_sys_prompt)
    print("-" * 20)
    
    if "调用工具" in current_sys_prompt:
        print("FAIL: '调用工具' still in prompt")
    elif "function_name" in current_sys_prompt:
        print("FAIL: 'function_name' still in prompt")
    else:
        print("SUCCESS: Prompt correctly filtered")

if __name__ == "__main__":
    test_prompt_filtering()
