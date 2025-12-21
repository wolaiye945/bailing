import sys
import os
sys.path.append(os.getcwd())

from plugins.task_manager import TaskManager
from plugins.registry import Action
import queue
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_task_manager_disabled():
    print("Starting test_task_manager_disabled...")
    
    # Test with enabled=False
    config_disabled = {
        "enabled": False,
        "functions_call_name": "plugins/function_calls_config.json",
        "aigc_manus_enabled": False
    }
    result_queue = queue.Queue()
    print("Initializing TaskManager with enabled=False...")
    tm_disabled = TaskManager(config_disabled, result_queue)
    
    print(f"TaskManager enabled: {tm_disabled.enabled}")
    assert tm_disabled.enabled is False
    
    functions = tm_disabled.get_functions()
    print(f"Functions count: {len(functions)}")
    assert len(functions) == 0
    
    print("Calling tool: get_day_of_week when disabled...")
    result = tm_disabled.tool_call("get_day_of_week", {})
    print(f"Action: {result.action}")
    assert result.action == Action.NOTFOUND
    
    # Test with enabled=True (default)
    config_enabled = {
        "enabled": True,
        "functions_call_name": "plugins/function_calls_config.json",
        "aigc_manus_enabled": False
    }
    print("\nInitializing TaskManager with enabled=True...")
    tm_enabled = TaskManager(config_enabled, result_queue)
    
    print(f"TaskManager enabled: {tm_enabled.enabled}")
    assert tm_enabled.enabled is True
    
    functions = tm_enabled.get_functions()
    print(f"Functions count: {len(functions)}")
    assert len(functions) > 0
    
    print("Calling tool: get_day_of_week when enabled...")
    # We don't want to actually run the tool if it has side effects, 
    # but get_day_of_week is safe.
    result = tm_enabled.tool_call("get_day_of_week", {})
    print(f"Action: {result.action}")
    # Action should NOT be NOTFOUND
    assert result.action != Action.NOTFOUND
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_task_manager_disabled()
