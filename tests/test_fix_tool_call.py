import sys
import os
sys.path.append(os.getcwd())

from plugins.task_manager import TaskManager
from plugins.registry import Action
import queue

def test_tool_call_wrapping():
    print("Starting test...")
    # Mock config
    config = {
        "functions_call_name": "plugins/function_calls_config.json",
        "aigc_manus_enabled": False
    }
    result_queue = queue.Queue()
    print("Initializing TaskManager...")
    tm = TaskManager(config, result_queue)
    
    print("Calling tool: get_day_of_week...")
    # Test get_day_of_week
    # It should return Action.REQLLM because the function itself returns it
    result = tm.tool_call("get_day_of_week", {})
    
    print(f"Action: {result.action}")
    print(f"Result: {result.result}")
    
    assert result.action == Action.REQLLM
    assert "星期" in result.result
    print("Test passed: get_day_of_week returns Action.REQLLM correctly.")
    print("Attempting to exit...")
    os._exit(0)

if __name__ == "__main__":
    test_tool_call_wrapping()
