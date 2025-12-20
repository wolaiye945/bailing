import json
import queue
import threading
import uuid
import os
from abc import ABC
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import argparse
import time

try:
    from static_ffmpeg import add_paths
    add_paths()
except ImportError:
    pass

from bailing import (
    recorder,
    player,
    asr,
    llm,
    tts,
    vad,
    memory
)
from bailing.dialogue import Message, Dialogue
from bailing.utils import is_interrupt, read_config, is_segment, extract_json_from_string, is_segment_sentence, remove_think_tags, format_think_sections
from bailing.prompt import sys_prompt

from plugins.registry import Action
from plugins.task_manager import TaskManager

logger = logging.getLogger(__name__)


class Robot(ABC):
    def __init__(self, config_file, websocket = None, loop = None):
        config = read_config(config_file)
        self.audio_queue = queue.Queue()

        self.recorder = recorder.create_instance(
            config["selected_module"]["Recorder"],
            config["Recorder"][config["selected_module"]["Recorder"]]
        )

        self.asr = asr.create_instance(
            config["selected_module"]["ASR"],
            config["ASR"][config["selected_module"]["ASR"]]
        )

        self.llm = llm.create_instance(
            config["selected_module"]["LLM"],
            config["LLM"][config["selected_module"]["LLM"]]
        )

        self.tts = tts.create_instance(
            config["selected_module"]["TTS"],
            config["TTS"][config["selected_module"]["TTS"]]
        )

        self.vad = vad.create_instance(
            config["selected_module"]["VAD"],
            config["VAD"][config["selected_module"]["VAD"]]
        )


        self.player = player.create_instance(
            config["selected_module"]["Player"],
            config["Player"][config["selected_module"]["Player"]]
        )

        memory_config = config.get("Memory")
        if memory_config and memory_config.get("enabled", True):
            self.memory = memory.Memory(memory_config)
            self.memory_text = self.memory.get_memory()
        else:
            self.memory = None
            self.memory_text = ""
        self.prompt = sys_prompt.replace("{memory}", self.memory_text).strip()

        self.vad_queue = queue.Queue()
        self.dialogue = Dialogue(config["Memory"]["dialogue_history_path"])
        self.dialogue.put(Message(role="system", content=self.prompt))

        # 保证tts是顺序的
        self.tts_queue = queue.Queue()
        # 初始化线程池
        self.executor = ThreadPoolExecutor(max_workers=10)

        self.vad_start = True

        # 打断相关配置
        self.INTERRUPT = config["interrupt"]
        self.silence_time_ms = int((1000 / 1000) * (16000 / 512))  # ms

        # 线程锁
        self.chat_lock = False

        # 事件用于控制程序退出
        self.stop_event = threading.Event()

        self.callback = None

        self.speech = []

        # 初始化单例
        #rag.Rag(config["Rag"])  # 第一次初始化

        self.task_queue = queue.Queue()
        self.task_manager = TaskManager(config.get("TaskManager"), self.task_queue)
        self.start_task_mode = config.get("StartTaskMode")

        if config["selected_module"]["Player"].lower().find("websocket") > -1:
            self.player.init(websocket, loop)
            self.listen_dialogue(self.player.send_messages)

    def listen_dialogue(self, callback):
        self.callback = callback

    def _stream_vad(self):
        def vad_thread():
            while not self.stop_event.is_set():
                try:
                    data = self.audio_queue.get()
                    vad_statue = self.vad.is_vad(data)
                    self.vad_queue.put({"voice": data, "vad_statue": vad_statue})
                except Exception as e:
                    logger.error(f"VAD 处理出错: {e}")
        consumer_audio = threading.Thread(target=vad_thread, daemon=True)
        consumer_audio.start()

    def _tts_priority(self):
        def priority_thread():
            while not self.stop_event.is_set():
                try:
                    future = self.tts_queue.get()
                    try:
                        tts_file = future.result(timeout=5)
                    except TimeoutError:
                        logger.error("TTS 任务超时")
                        continue
                    except Exception as e:
                        logger.error(f"TTS 任务出错: {e}")
                        continue
                    if tts_file is None:
                        continue
                    logger.debug(f"_tts_priority: 准备播放 {tts_file}")
                    self.player.play(tts_file)
                except Exception as e:
                    logger.error(f"tts_priority priority_thread: {e}")
        tts_priority = threading.Thread(target=priority_thread, daemon=True)
        tts_priority.start()

    def interrupt_playback(self):
        """中断当前的语音播放"""
        logger.info("Interrupting current playback.")
        self.player.stop()

    def shutdown(self):
        """关闭所有资源，确保程序安全退出"""
        logger.info("Shutting down Robot...")
        self.stop_event.set()
        self.executor.shutdown(wait=True)
        self.recorder.stop_recording()
        self.player.shutdown()
        logger.info("Shutdown complete.")

    def start_recording_and_vad(self):
        # 开始监听语音流
        self.recorder.start_recording(self.audio_queue)
        logger.info("Started recording.")
        # vad 实时识别
        self._stream_vad()
        # tts优先级队列
        self._tts_priority()

    def _duplex(self):
        # 处理识别结果
        data = self.vad_queue.get()
        # 识别到vad开始
        if self.vad_start:
            self.speech.append(data)
        vad_status = data.get("vad_statue")
        # 空闲的时候，取出耗时任务进行播放
        if not self.task_queue.empty() and  not self.vad_start and vad_status is None \
                and not self.player.get_playing_status() and self.chat_lock is False:
            result = self.task_queue.get()
            future = self.executor.submit(self.speak_and_play, result.response)
            self.tts_queue.put(future)

        """ 语音唤醒
        if time.time() - self.start_time>=60:
            self.silence_status = True

        if self.silence_status:
            return
        """
        if vad_status is None:
            return
        if "start" in vad_status:
            if self.player.get_playing_status() or self.chat_lock is True:  # 正在播放，打断场景
                if self.INTERRUPT:
                    self.chat_lock = False
                    self.interrupt_playback()
                    self.vad_start = True
                    self.speech.append(data)
                else:
                    return
            else:  # 没有播放，正常
                self.vad_start = True
                self.speech.append(data)
        elif "end" in vad_status and len(self.speech) > 0:
            try:
                logger.debug(f"语音包的长度：{len(self.speech)}")
                self.vad_start = False
                voice_data = [d["voice"] for d in self.speech]
                text, tmpfile = self.asr.recognizer(voice_data)
                self.speech = []
            except Exception as e:
                self.vad_start = False
                self.speech = []
                logger.error(f"ASR识别出错: {e}")
                return
            if not text.strip():
                logger.debug("识别结果为空，跳过处理。")
                return

            logger.debug(f"ASR识别结果: {text}")
            if self.callback:
                self.callback({"role": "user", "content": str(text)})
            self.executor.submit(self.chat, text)
        return True

    def run(self):
        try:
            self.start_recording_and_vad()  # 监听语音流
            while not self.stop_event.is_set():
                self._duplex()  # 双工处理
        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt. Exiting...")
        finally:
            self.shutdown()

    def speak_and_play(self, text):
        if text is None or len(text)<=0:
            logger.info(f"无需tts转换，query为空，{text}")
            return None
        tts_file = self.tts.to_tts(text)
        if tts_file is None:
            logger.error(f"tts转换失败，{text}")
            return None
        logger.debug(f"TTS 文件生成完毕{self.chat_lock}")
        #if self.chat_lock is False:
        #    return None
        # 开始播放
        # self.player.play(tts_file)
        return tts_file

    def chat_tool(self, query):
        # 打印逐步生成的响应内容
        start = 0
        try:
            start_time = time.time()  # 记录开始时间
            llm_responses = self.llm.response_call(self.dialogue.get_llm_dialogue(), functions_call=self.task_manager.get_functions())
        except Exception as e:
            #self.chat_lock = False
            logger.error(f"LLM 处理出错 {query}: {e}")
            return []

        tool_call_flag = False
        response_message = []
        # tool call 参数
        function_name = None
        function_id = None
        function_arguments = ""
        content_arguments = ""
        
        for chunk in llm_responses:
            content, tools_call = chunk
            
            # 尽早检测工具调用
            if tools_call is not None:
                tool_call_flag = True
                if tools_call[0].id is not None:
                    function_id = tools_call[0].id
                if tools_call[0].function.name is not None:
                    function_name = tools_call[0].function.name
                if tools_call[0].function.arguments is not None:
                    function_arguments += tools_call[0].function.arguments
            
            if content is not None and len(content) > 0:
                # 检查是否包含代码块标记，这通常意味着工具调用（对于某些模型）
                if "```" in content or (len(response_message) == 0 and content.strip().startswith("{")):
                    tool_call_flag = True
                
                if tool_call_flag:
                    content_arguments += content
                else:
                    response_message.append(content)
                    response_message_concat = "".join(response_message)
                    
                    # 过滤掉思考部分再进行分句
                    clean_response_concat = remove_think_tags(response_message_concat)
                    
                    # 检查 clean_response_concat 是否包含可能触发工具调用的关键字
                    if "```" in clean_response_concat:
                        tool_call_flag = True
                        # 将之前错误识别为文本的部分移到 content_arguments
                        content_arguments = clean_response_concat[start:]
                        continue

                    end_time = time.time()
                    logger.debug(f"大模型返回时间: {end_time - start_time} 秒, token={content}")
                    
                    flag_segment, index_segment = is_segment_sentence(clean_response_concat, start)
                    if flag_segment:
                        segment_text = clean_response_concat[start:index_segment + 1]
                        
                        # 再次检查 segment_text 是否包含工具调用标记
                        if "```" in segment_text:
                            tool_call_flag = True
                            content_arguments = segment_text
                            continue

                        if len(segment_text) <= max(2, start):
                            continue
                            
                        future = self.executor.submit(self.speak_and_play, segment_text)
                        self.tts_queue.put(future)
                        start = index_segment + 1

        if not tool_call_flag:
            clean_response_concat = remove_think_tags("".join(response_message))
            if start < len(clean_response_concat):
                segment_text = clean_response_concat[start:]
                # 确保最后一部分也不是工具调用
                if "```" not in segment_text:
                    future = self.executor.submit(self.speak_and_play, segment_text)
                    self.tts_queue.put(future)
        else:
            # 处理函数调用
            if function_id is None:
                json_str = extract_json_from_string(content_arguments)
                if json_str is not None:
                    try:
                        content_arguments_json = json.loads(json_str)
                        function_name = content_arguments_json.get("function_name")
                        function_arguments = content_arguments_json.get("args", {})
                        function_id = str(uuid.uuid4().hex)
                    except Exception as e:
                        logger.error(f"解析工具调用 JSON 失败: {e}")
                        return response_message
                else:
                    return response_message
            
            logger.info(f"工具调用: name={function_name}, args={function_arguments}")
            
            # 调用工具
            result = self.task_manager.tool_call(function_name, function_arguments)
            if result.action == Action.NOTFOUND:
                logger.error(f"没有找到函数: {function_name}")
                return response_message
            elif result.action == Action.NONE:
                return response_message
            elif result.action == Action.RESPONSE:
                future = self.executor.submit(self.speak_and_play, result.response)
                self.tts_queue.put(future)
                response_message.append(f"\n工具执行结果: {result.response}")
                return response_message
            elif result.action == Action.REQLLM:
                # 添加工具内容到对话上下文
                self.dialogue.put(Message(role='assistant',
                                          tool_calls=[{"id": function_id, "function": {"arguments": json.dumps(function_arguments ,ensure_ascii=False),
                                                                                       "name": function_name},
                                                       "type": 'function', "index": 0}]))
                self.dialogue.put(Message(role="tool", tool_call_id=function_id, content=result.result))
                # 递归调用以处理后续回复
                sub_response = self.chat_tool(query)
                if sub_response:
                    response_message.extend(sub_response)
            elif result.action == Action.ADDSYSTEM:
                self.dialogue.put(Message(**result.result))
                return response_message
            elif result.action == Action.ADDSYSTEMSPEAK:
                self.dialogue.put(Message(role='assistant',
                                          tool_calls=[{"id": function_id, "function": {
                                              "arguments": json.dumps(function_arguments, ensure_ascii=False),
                                              "name": function_name},
                                                       "type": 'function', "index": 0}]))
                self.dialogue.put(Message(role="tool", tool_call_id=function_id, content=result.response))
                self.dialogue.put(Message(**result.result))
                self.dialogue.put(Message(role="user", content="ok"))
                sub_response = self.chat_tool(query)
                if sub_response:
                    response_message.extend(sub_response)
            else:
                logger.error(f"未知的 Action 类型: {result.action}")
        
        return response_message

    def chat(self, query):
        self.dialogue.put(Message(role="user", content=query))
        response_message = []
        # futures = []
        start = 0
        self.chat_lock = True
        if self.start_task_mode:
            response_message = self.chat_tool(query)
        else:
            # 提交 LLM 任务
            try:
                start_time = time.time()  # 记录开始时间
                llm_responses = self.llm.response(self.dialogue.get_llm_dialogue())
            except Exception as e:
                self.chat_lock = False
                logger.error(f"LLM 处理出错 {query}: {e}")
                return None
            # 提交 TTS 任务到线程池
            for content in llm_responses:
                response_message.append(content)
                response_message_concat = "".join(response_message)
                
                # 过滤掉思考部分再进行分句
                clean_response_concat = remove_think_tags(response_message_concat)
                
                end_time = time.time()  # 记录结束时间
                logger.debug(f"大模型返回时间时间tool: {end_time - start_time} 秒, 生成token={content}")
                
                flag_segment, index_segment = is_segment_sentence(clean_response_concat, start)
                logger.debug(
                    f"大模型返回时间时间tool: flag_segment={flag_segment} 秒, index_segment={index_segment}")
                if flag_segment:
                    segment_text = clean_response_concat[start:index_segment + 1]
                    # 为了保证语音的连贯，至少2个字才转tts

                    if len(segment_text) <= max(2, start):
                        continue
                    future = self.executor.submit(self.speak_and_play, segment_text)
                    self.tts_queue.put(future)
                    # futures.append(future)
                    start = index_segment + 1  # len(response_message)

            # 处理剩余的响应
            clean_response_concat = remove_think_tags("".join(response_message))
            if start < len(clean_response_concat):
                segment_text = clean_response_concat[start:]
                future = self.executor.submit(self.speak_and_play, segment_text)
                self.tts_queue.put(future)
            # 等待所有 TTS 任务完成
            """
            for future in futures:
                try:
                    playing = future.result(timeout=5)
                except TimeoutError:
                    logger.error("TTS 任务超时")
                except Exception as e:
                    logger.error(f"TTS 任务出错: {e}")
            """
        self.chat_lock = False
        # 更新对话
        full_response = "".join(response_message)
        formatted_response = format_think_sections(full_response)
        
        if self.callback:
            self.callback({"role": "assistant", "content": formatted_response})
        self.dialogue.put(Message(role="assistant", content=full_response))
        self.dialogue.dump_dialogue()
        logger.debug(json.dumps(self.dialogue.get_llm_dialogue(), indent=4, ensure_ascii=False))
        return True


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="百聆机器人")

    # Add arguments
    parser.add_argument('config_path', type=str, help="配置文件", default=None)

    # Parse arguments
    args = parser.parse_args()
    config_path = args.config_path

    # 创建 Robot 实例并运行
    robot = Robot(config_path)
    robot.run()
