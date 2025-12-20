import sys
import os
import queue
import logging

# Set up logging to stdout
logging.basicConfig(level=logging.INFO)

# Add current directory to path
sys.path.append(os.getcwd())

from plugins.task_manager import TaskManager
from bailing.utils import read_config

def test():
    config_path = "config/config.yaml"
    config = read_config(config_path)
    
    result_queue = queue.Queue()
    task_manager = TaskManager(config.get("TaskManager"), result_queue)
    
    print("\n--- Loaded Functions ---")
    for func in task_manager.get_functions():
        print(f"Function: {func['function']['name']}")
    
    from plugins.registry import function_registry
    print("\n--- Registered Functions in Registry ---")
    for name in function_registry.keys():
        print(f"Registered: {name}")
    
    print("\n--- Test Complete ---")
    os._exit(0)

if __name__ == "__main__":
    test()
