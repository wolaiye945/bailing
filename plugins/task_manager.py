import logging
import importlib
import pkgutil
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from plugins.registry import function_registry, Action, ActionResponse, ToolType
from bailing.utils import read_json_file


logger = logging.getLogger(__name__)


def auto_import_modules(package_name, skip_modules=None):
    """
    自动导入指定包内的所有模块。

    Args:
        package_name (str): 包的名称，如 'functions'。
        skip_modules (list): 需要跳过的模块名称列表。
    """
    if skip_modules is None:
        skip_modules = []
    
    # 获取包的路径
    package = importlib.import_module(package_name)
    package_path = package.__path__

    # 遍历包内的所有模块
    for _, module_name, _ in pkgutil.iter_modules(package_path):
        if module_name in skip_modules:
            logger.info(f"跳过模块 '{module_name}'")
            continue
            
        # 导入模块
        try:
            full_module_name = f"{package_name}.{module_name}"
            importlib.import_module(full_module_name)
            logger.info(f"模块 '{full_module_name}' 已加载")
        except Exception as e:
            logger.error(f"模块 '{full_module_name}' 加载失败: {e}", exc_info=True)

# Remove top-level auto-import
# auto_import_modules('plugins.functions')


class TaskManager:
    def __init__(self, config, result_queue: queue.Queue):
        aigc_manus_enabled = config.get("aigc_manus_enabled", False)
        
        # 根据配置决定要跳过的模块
        skip_list = []
        if not aigc_manus_enabled:
            skip_list.append('aigc_manus')
            
        # 自动导入 'functions' 包中的模块
        auto_import_modules('plugins.functions', skip_modules=skip_list)
        
        self.functions = read_json_file(config.get("functions_call_name"))
        if not aigc_manus_enabled:
            self.functions = [item for item in self.functions if item["function"]["name"] != 'aigc_manus']
        self.task_queue = queue.Queue()
        # 初始化线程池
        self.task_executor = ThreadPoolExecutor(max_workers=10)
        self.result_queue = result_queue

    def get_functions(self):
        return self.functions

    def process_task(self):
        def task_thread():
            while True:
                try:
                    # 从队列中取出已完成的任务
                    while not self.task_queue.empty():
                        future = self.task_queue.get()
                        if future.done():  # 检查任务是否完成
                            result = future.result()  # 获取任务结果
                            self.result_queue.put(result)
                        else:
                            self.task_queue.put(future)  # 如果没有完成，放回队列
                        time.sleep(1)
                except Exception as e:
                    logger.error(f"task_thread 处理出错: {e}")
                time.sleep(2)
        consumer_task = threading.Thread(target=task_thread, daemon=True)
        consumer_task.start()

    @staticmethod
    def call_function(func_name, *args, **kwargs):
        """
        通用函数调用方法

        :param func_name: 函数名称 (str)
        :param args: 函数的位置参数
        :param kwargs: 函数的关键字参数
        :return: 函数调用的结果
        """
        try:
            # 从注册器中获取函数
            if func_name in function_registry:
                func = function_registry[func_name]
                # 调用函数，并传递参数
                result = func(*args, **kwargs)
                return result
            else:
                raise ValueError(f"函数 '{func_name}' 未注册！")
        except Exception as e:
            return f"调用函数 '{func_name}' 时出错：{str(e)}"

    def tool_call(self, func_name, func_args) -> ActionResponse:
        if func_name not in function_registry:
            return ActionResponse(action=Action.NOTFOUND, result="没有找到相应函数", response=None)
        
        # 确保 func_args 是 dict 类型
        if isinstance(func_args, str):
            try:
                import json
                func_args = json.loads(func_args)
            except Exception as e:
                logger.error(f"解析 func_args 失败: {e}, func_args={func_args}")
                func_args = {}
        
        if func_args is None:
            func_args = {}
            
        if not isinstance(func_args, dict):
            logger.warning(f"func_args 不是字典类型，强制转换为为空字典: {type(func_args)}")
            func_args = {}

        func = function_registry[func_name]
        result = self.call_function(func_name, **func_args)
        
        # 如果结果本身就是 ActionResponse，直接返回
        # 使用类名判断，防止因为不同导入路径导致的 isinstance 失败
        if result.__class__.__name__ == 'ActionResponse':
            return result
            
        # 否则根据 ToolType 包装结果
        if func.action == ToolType.NONE:
            future = self.task_executor.submit(self.call_function, func_name, **func_args)
            self.task_queue.put(future)
            return ActionResponse(action=Action.NONE, result=None, response=None)
        elif func.action == ToolType.WAIT:
            return ActionResponse(action=Action.RESPONSE, result=result, response=result)
        elif func.action == ToolType.SCHEDULER:
            return ActionResponse(action=Action.RESPONSE, result=result, response=result)
        elif func.action == ToolType.TIME_CONSUMING:
            future = self.task_executor.submit(self.call_function, func_name, **func_args)
            self.task_queue.put(future)
            return ActionResponse(action=Action.RESPONSE, result=None, response="您好，正在查询信息中，一会查询完我会告诉你哟")
        elif func.action == ToolType.ADD_SYS_PROMPT:
            return ActionResponse(action=Action.ADDSYSTEMSPEAK, result=result, response=None)
        else:
            return ActionResponse(action=Action.RESPONSE, result=result, response=result)

if __name__ == "__main__":
    pass