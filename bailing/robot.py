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
    memory,
    noise_reduction
)
from bailing.dialogue import Message, Dialogue
from bailing.utils import is_interrupt, read_config, is_segment, extract_json_from_string, is_segment_sentence, remove_think_tags, format_think_sections
from bailing.prompt import sys_prompt

from plugins.registry import Action
from plugins.task_manager import TaskManager

logger = logging.getLogger(__name__)


class Robot(ABC):
    def __init__(self, config_file, websocket = None, loop = None, user_info = None):
        logger.info(f"Robot 正在从 {config_file} 初始化...")
        # 确保 user_info 是字典格式
        if isinstance(user_info, dict):
            self.user_info = user_info
        elif isinstance(user_info, str):
            self.user_info = {"username": user_info, "role": "user"}
        else:
            self.user_info = {"username": "default", "role": "user"}
            
        config = read_config(config_file)
        self.audio_queue = queue.Queue()

        logger.info(f"初始化 Recorder (User: {self.user_info['username']})...")
        self.recorder = recorder.create_instance(
            config["selected_module"]["Recorder"],
            config["Recorder"][config["selected_module"]["Recorder"]]
        )

        logger.info(f"初始化 ASR (User: {self.user_info['username']})...")
        self.asr = asr.create_instance(
            config["selected_module"]["ASR"],
            config["ASR"][config["selected_module"]["ASR"]]
        )

        logger.info(f"初始化 LLM (User: {self.user_info['username']})...")
        self.llm = llm.create_instance(
            config["selected_module"]["LLM"],
            config["LLM"][config["selected_module"]["LLM"]]
        )

        logger.info(f"初始化 TTS (User: {self.user_info['username']})...")
        self.tts = tts.create_instance(
            config["selected_module"]["TTS"],
            config["TTS"][config["selected_module"]["TTS"]]
        )

        logger.info("初始化 VAD...")
        self.vad = vad.create_instance(
            config["selected_module"]["VAD"],
            config["VAD"][config["selected_module"]["VAD"]]
        )

        logger.info("初始化 Player...")
        self.player = player.create_instance(
            config["selected_module"]["Player"],
            config["Player"][config["selected_module"]["Player"]]
        )

        logger.info("初始化 Noise Reduction...")
        self.nr = noise_reduction.create_instance(
            config.get("NoiseReduction", {"enabled": False})
        )

        # 事件用于控制程序退出
        self.stop_event = threading.Event()
        # 线程锁与会话管理
        self.chat_lock = False
        self.chat_session_id = 0

        # 保证tts是顺序的
        self.tts_queue = queue.Queue()
        # 初始化线程池
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # 启动 TTS 优先级队列处理线程
        self._tts_priority()

        self.task_queue = queue.Queue()
        self.task_manager = TaskManager(config.get("TaskManager"), self.task_queue)
        self.start_task_mode = config.get("StartTaskMode")
        
        # 初始化 EOQ 配置
        self.eoq_config = config.get("EOQ", {"enabled": False})

        memory_config = config.get("Memory", {}).copy()
        if memory_config and memory_config.get("enabled", True):
            # 为不同用户隔离 memory 和对话历史路径
            if self.user_info['username'] != "default":
                user_path = self.user_info['username']
                
                # 优化对话历史路径
                orig_history_path = memory_config.get("dialogue_history_path", "tmp/dialogue")
                memory_config["dialogue_history_path"] = os.path.join(orig_history_path, user_path)
                
                # 优化 memory 文件路径
                orig_memory_file = memory_config.get("memory_file", "tmp/memory.json")
                memory_dir = os.path.dirname(orig_memory_file)
                memory_base = os.path.basename(orig_memory_file)
                memory_config["memory_file"] = os.path.join(memory_dir, user_path, memory_base)
                
                # 确保目录存在
                os.makedirs(os.path.dirname(memory_config["memory_file"]), exist_ok=True)
                os.makedirs(memory_config["dialogue_history_path"], exist_ok=True)

            self.memory = memory.Memory(memory_config)
            self.memory_text = self.memory.get_memory()
        else:
            self.memory = None
            self.memory_text = ""
        
        current_sys_prompt = sys_prompt
        # 如果任务管理器未启用，或者未开启任务模式，则移除工具调用相关的提示词
        if not self.task_manager.enabled or not self.start_task_mode:
            # 移除包含 "调用工具" 或 "function_name" 的行，这些通常与工具调用说明相关
            lines = current_sys_prompt.split('\n')
            filtered_lines = [line for line in lines if "调用工具" not in line and "function_name" not in line]
            current_sys_prompt = '\n'.join(filtered_lines)

        self.prompt = current_sys_prompt.replace("{memory}", self.memory_text).strip()

        self.vad_queue = queue.Queue()
        self.max_history = config.get("Memory", {}).get("max_history", 15)
        self.dialogue = Dialogue(memory_config.get("dialogue_history_path", "tmp/dialogue"))
        self.dialogue.put(Message(role="system", content=self.prompt))

        self.vad_start = True

        # 打断相关配置
        self.INTERRUPT = config["interrupt"]
        self.echo_config = config.get("echo_cancellation", {"enabled": False})
        self.silence_time_ms = int((1000 / 1000) * (16000 / 512))  # ms

        self.callback = None

        self.speech = []

        # 初始化单例
        #rag.Rag(config["Rag"])  # 第一次初始化

        if config["selected_module"]["Player"].lower().find("websocket") > -1:
            self.player.init(websocket, loop)
            self.listen_dialogue(self.player.send_messages)

    def _is_meaningful_query(self, query):
        """
        智能意图判断 (EOQ): 判断 query 是否是一个有意义的对话输入
        返回 True 表示有意义，需要处理；返回 False 表示无意义，应忽略。
        """
        if not self.eoq_config.get("enabled", True):
            return True
            
        if not query:
            return False
            
        clean_query = query.strip().replace(" ", "").replace("。", "").replace("，", "").replace("？", "").replace("！", "")
        
        # 1. 检查是否在常用指令列表中
        common_cmds = self.eoq_config.get("common_cmds", [])
        if clean_query in common_cmds:
            return True
            
        # 2. 检查长度
        min_len = self.eoq_config.get("min_length", 2)
        if len(clean_query) < min_len:
            logger.info(f"EOQ: 输入太短 ({query})，且不在常用指令中，判定为误触发")
            return False
            
        # 3. 检查是否全是语气词/填充词
        filler_words = self.eoq_config.get("filler_words", [])
        is_all_fillers = True
        for char in clean_query:
            if char not in filler_words:
                is_all_fillers = False
                break
        if is_all_fillers:
            logger.info(f"EOQ: 输入仅含填充词 ({query})，判定为误触发")
            return False
            
        # 4. 上下文感知 (进阶)：如果机器人正在等待输入，则更倾向于认为是有效的
        # 这里可以根据 self.player.get_playing_status() 或对话历史状态来判断
        
        return True

    def listen_dialogue(self, callback):
        self.callback = callback

    def _stream_vad(self):
        def vad_thread():
            while not self.stop_event.is_set():
                try:
                    # 使用 timeout 允许检查 stop_event
                    data = self.audio_queue.get(timeout=1.0)
                    
                    # 算法降噪处理
                    if self.nr.enabled:
                        data = self.nr.process(data)
                    
                    vad_statue = self.vad.is_vad(data)
                    self.vad_queue.put({"voice": data, "vad_statue": vad_statue})
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"VAD 处理出错: {e}")
        consumer_audio = threading.Thread(target=vad_thread, daemon=True)
        consumer_audio.start()

    def _tts_priority(self):
        logger.info("正在启动 TTS 优先级处理线程...")
        def priority_thread():
            logger.info("TTS 优先级处理线程已开始运行")
            while not self.stop_event.is_set():
                try:
                    # 使用 timeout 允许检查 stop_event
                    logger.debug("正在等待 TTS 任务...")
                    task = self.tts_queue.get(timeout=1.0)
                    if isinstance(task, tuple) and len(task) == 2:
                        session_id, future = task
                    else:
                        # 兼容旧版本，如果队列里只有 future
                        session_id = self.chat_session_id
                        future = task

                    # 检查会话是否仍然有效
                    if session_id != self.chat_session_id:
                        logger.info(f"忽略过期会话的 TTS 任务: session_id={session_id}, current={self.chat_session_id}")
                        continue

                    logger.info(f"获取到 TTS 任务，当前队列大小: {self.tts_queue.qsize()}")
                    try:
                        # 设置较长的超时（如 60s），防止 TTS 任务永久卡死导致整个播放队列阻塞
                        # 同时保留按序处理的特性
                        logger.info("正在等待 TTS 任务执行结果...")
                        tts_file = future.result(timeout=60) 
                        logger.info(f"TTS 任务执行完成，生成文件: {tts_file}")
                    except TimeoutError:
                        logger.error("TTS 任务执行超过 60s，强制跳过当前片段以恢复队列")
                        continue
                    except Exception as e:
                        logger.error(f"TTS 任务出错: {e}")
                        continue

                    # 再次检查会话，因为 future.result() 可能等了很久
                    if session_id != self.chat_session_id:
                        logger.info(f"TTS 任务完成后发现会话已失效，跳过播放: session_id={session_id}")
                        continue

                    if tts_file is None:
                        logger.warning("TTS 文件为 None，跳过播放")
                        continue
                    logger.info(f"_tts_priority: 准备播放 {tts_file}")
                    self.player.play(tts_file)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"tts_priority priority_thread: {e}")
            logger.info("TTS 优先级处理线程已退出")
        tts_priority = threading.Thread(target=priority_thread, daemon=True)
        tts_priority.start()

    def interrupt_playback(self):
        """中断当前的语音播放"""
        if not self.INTERRUPT:
            return
        logger.info(f"!!! 打断播放器 !!! (User: {self.user_info['username']}, Session: {self.chat_session_id})")
        self.player.stop()
        # 清空 tts 队列，防止旧的 segment 在打断后继续播放
        with self.tts_queue.mutex:
            self.tts_queue.queue.clear()
        self.chat_lock = False # 重置 chat_lock，允许打断 LLM 生成
        self.chat_session_id += 1 # 增加会话 ID，让旧的 chat 线程失效

    def shutdown(self):
        """关闭所有资源，确保程序安全退出"""
        logger.info(f"正在关闭 Robot 实例: {id(self)}")
        self.stop_event.set()
        # 不要 wait=True，避免长时间阻塞
        self.executor.shutdown(wait=False)
        self.recorder.stop_recording()
        self.player.shutdown()
        logger.info(f"Robot 实例 {id(self)} 已完成关闭流程")

    def start_recording_and_vad(self):
        # 开始监听语音流
        self.recorder.start_recording(self.audio_queue)
        logger.info("Started recording.")
        # vad 实时识别
        self._stream_vad()

    def _duplex(self):
        # 处理识别结果
        try:
            # 使用 timeout 允许检查 stop_event
            data = self.vad_queue.get(timeout=1.0)
        except queue.Empty:
            return
            
        # 动态调整 VAD 阈值以抑制回声
        if self.echo_config.get("enabled"):
            if self.player.get_playing_status():
                self.vad.set_threshold(self.echo_config.get("threshold_playing", 0.95))
            else:
                self.vad.set_threshold(self.vad.original_threshold)

        # 识别到vad开始
        if self.vad_start:
            self.speech.append(data)
        
        vad_status = data.get("vad_statue")
        
        # 只有在有 vad_status 时才记录 debug 日志，避免太多无用信息
        if vad_status:
            logger.debug(f"VAD 状态更新: {vad_status}")

        # 空闲的时候，取出耗时任务进行播放
        if not self.task_queue.empty() and  not self.vad_start and vad_status is None \
                and not self.player.get_playing_status() and self.chat_lock is False:
            result = self.task_queue.get()
            future = self.executor.submit(self.speak_and_play, result.response)
            self.tts_queue.put((self.chat_session_id, future))

        """ 语音唤醒
        if time.time() - self.start_time>=60:
            self.silence_status = True

        if self.silence_status:
            return
        """
        if vad_status is None:
            return
        if "start" in vad_status:
            if hasattr(self.player, 'send_status'):
                self.player.send_status("listening")
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
        elif "end" in vad_status:
            if vad_status.get("cancel"):
                logger.info("VAD 判定为噪声，取消本次识别")
                self.vad_start = False
                self.speech = []
                if hasattr(self.player, 'send_status'):
                    self.player.send_status("idle")
                return

            if len(self.speech) > 0:
                logger.info(f"检测到说话结束，异步启动 ASR 识别 (语音包长度: {len(self.speech)})")
                self.vad_start = False
                self.vad.reset_states()  # 重置 VAD 状态
                voice_data = [d["voice"] for d in self.speech]
                self.speech = []
                
                def asr_and_chat(data):
                    try:
                        if hasattr(self.player, 'send_status'):
                            self.player.send_status("processing")
                        logger.info(f"开始 ASR 识别 (User: {self.user_info['username']})...")
                        text, tmpfile = self.asr.recognizer(data, username=self.user_info['username'])
                        if text is None or not text.strip():
                            logger.info("ASR 识别结果为空，跳过处理。")
                            if hasattr(self.player, 'send_status'):
                                self.player.send_status("idle")
                            return
                        
                        logger.info(f"ASR 识别成功: {text}")
                        
                        # 智能意图判断 (EOQ)
                        if not self._is_meaningful_query(text):
                            if hasattr(self.player, 'send_status'):
                                self.player.send_status("idle")
                            return

                        if hasattr(self.player, 'send_status'):
                            self.player.send_status("thinking")
                        if self.callback:
                            self.callback({"role": "user", "content": str(text)})
                        self.chat(text)
                    except Exception as e:
                        logger.error(f"ASR/Chat 异步处理出错: {e}", exc_info=True)
                        if hasattr(self.player, 'send_status'):
                            self.player.send_status("idle")
                    finally:
                        logger.info("ASR/Chat 异步处理线程结束")

                self.executor.submit(asr_and_chat, voice_data)
            else:
                logger.debug("检测到说话结束，但语音包为空，忽略")
                self.vad_start = False
                self.vad.reset_states()
        return True

    def run(self):
        logger.info("Robot 运行线程已启动")
        try:
            self.start_recording_and_vad()  # 监听语音流
            logger.info("语音录音和 VAD 已启动")
            while not self.stop_event.is_set():
                try:
                    self._duplex()  # 双工处理
                except Exception as e:
                    logger.error(f"Robot _duplex 循环出错: {e}", exc_info=True)
                    time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("收到 KeyboardInterrupt，正在退出...")
        except Exception as e:
            logger.error(f"Robot 运行线程崩溃: {e}", exc_info=True)
        finally:
            logger.info("Robot 运行线程即将结束，执行清理...")
            self.shutdown()

    def speak_and_play(self, text):
        if text is None or len(text)<=0:
            logger.info(f"无需tts转换，query为空，{text}")
            return None
        logger.info(f"--- speak_and_play START: {text[:50]}... ---")
        try:
            tts_file = self.tts.to_tts(text, username=self.user_info['username'])
            logger.info(f"--- speak_and_play TTS DONE: {tts_file} ---")
        except Exception as e:
            logger.error(f"TTS 转换抛出异常: {e}", exc_info=True)
            return None
            
        if tts_file is None:
            logger.error(f"tts转换失败，{text}")
            return None
        logger.info(f"TTS 文件生成完毕: {tts_file}, 当前 chat_lock: {self.chat_lock}")
        # 开始播放
        return tts_file

    def chat_tool(self, query, depth=0, session_id=None):
        # 智能意图判断 (EOQ)
        if not self._is_meaningful_query(query):
            self.chat_lock = False
            return []

        # 限制递归深度，防止死循环
        if depth > 5:
            logger.error("达到最大工具调用深度，停止递归")
            return ["抱歉，我陷入了处理循环，请尝试换个方式提问。"]

        if session_id is None:
            session_id = self.chat_session_id

        # 打印逐步生成的响应内容
        start = 0
        try:
            start_time = time.time()  # 记录开始时间
            llm_responses = self.llm.response_call(self.dialogue.get_llm_dialogue(max_history=self.max_history), functions_call=self.task_manager.get_functions())
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
            if self.stop_event.is_set():
                logger.info("检测到 stop_event，停止 chat_tool LLM 响应迭代")
                break
            
            if not self.chat_lock or (session_id is not None and session_id != self.chat_session_id):
                logger.info(f"检测到会话失效 (lock={self.chat_lock}, session={session_id}/{self.chat_session_id})，停止 chat_tool LLM 响应迭代")
                break

            content, tools_call = chunk
            
            if len(response_message) == 0 and not tool_call_flag:
                if hasattr(self.player, 'send_status'):
                    self.player.send_status("responding")

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
                # 检查是否包含代码块标记或工具箱标记，这通常意味着工具调用（对于某些模型）
                # 如果已经有了原生的 tools_call，就不再尝试从 content 中解析 JSON
                if tools_call is None and ("```" in content or "<|begin_of_box|>" in content or (len(response_message) == 0 and content.strip().startswith("{"))):
                    tool_call_flag = True
                
                if tool_call_flag and tools_call is None:
                    content_arguments += content
                else:
                    response_message.append(content)
                    response_message_concat = "".join(response_message)
                    
                    # 过滤掉思考部分再进行分句
                    clean_response_concat = remove_think_tags(response_message_concat)
                    
                    # 检查 clean_response_concat 是否包含可能触发工具调用的关键字
                    if "```" in clean_response_concat or "<|begin_of_box|>" in clean_response_concat:
                        tool_call_flag = True
                        # 将之前错误识别为文本的部分移到 content_arguments
                        content_arguments = clean_response_concat[start:]
                        continue

                    end_time = time.time()
                    logger.debug(f"大模型返回时间: {end_time - start_time} 秒, token={content}")
                    
                    # 循环查找所有可能的分句，处理一个 chunk 中包含多个句子的情况
                    while True:
                        flag_segment, index_segment = is_segment_sentence(clean_response_concat, start)
                        if flag_segment:
                            segment_text = clean_response_concat[start:index_segment + 1].strip()
                            
                            # 再次检查 segment_text 是否包含工具调用标记
                            if "```" in segment_text or "<|begin_of_box|>" in segment_text:
                                tool_call_flag = True
                                content_arguments = segment_text
                                break # 跳出 while True 循环，外层会处理 tool_call_flag
    
                            if len(segment_text) >= 1:
                                logger.info(f"[{session_id}] 识别到分句 (start={start}, end={index_segment}): {segment_text}")
                                future = self.executor.submit(self.speak_and_play, segment_text)
                                self.tts_queue.put((session_id, future))
                                logger.debug(f"[{session_id}] 分句已入队，当前队列大小: {self.tts_queue.qsize()}")
                            start = index_segment + 1
                        else:
                            break
                    
                    if tool_call_flag:
                        continue

        if not tool_call_flag:
            clean_response_concat = remove_think_tags("".join(response_message))
            if start < len(clean_response_concat):
                segment_text = clean_response_concat[start:].strip()
                # 确保最后一部分也不是工具调用
                if segment_text and "```" not in segment_text and "<|begin_of_box|>" not in segment_text:
                    logger.info(f"[{session_id}] 识别到最后分句 (start={start}): {segment_text}")
                    future = self.executor.submit(self.speak_and_play, segment_text)
                    self.tts_queue.put((session_id, future))
                    logger.debug(f"[{session_id}] 最后分句已入队")
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
                if result.response:
                    future = self.executor.submit(self.speak_and_play, result.response)
                    self.tts_queue.put((session_id, future))
                    response_message.append(result.response)
                return response_message
            elif result.action == Action.REQLLM:
                # 添加工具调用和结果到对话历史，但不添加到 response_message（避免重复显示思考过程）
                # 注意：这里我们选择不清空 response_message，因为之前的思考过程可能对用户有意义
                self.dialogue.put(Message(role='assistant',
                                          content="".join(response_message),
                                          tool_calls=[{"id": function_id, "function": {"arguments": json.dumps(function_arguments ,ensure_ascii=False) if isinstance(function_arguments, dict) else function_arguments,
                                                                                       "name": function_name},
                                                       "type": 'function', "index": 0}]))
                self.dialogue.put(Message(role="tool", tool_call_id=function_id, content=result.result))
                # 递归调用以处理后续回复
                sub_response = self.chat_tool(query, depth=depth + 1, session_id=session_id)
                if sub_response:
                    response_message.extend(sub_response)
            elif result.action == Action.ADDSYSTEM:
                self.dialogue.put(Message(**result.result))
                return response_message
            elif result.action == Action.ADDSYSTEMSPEAK:
                self.dialogue.put(Message(role='assistant',
                                          content="".join(response_message),
                                          tool_calls=[{"id": function_id, "function": {
                                              "arguments": json.dumps(function_arguments, ensure_ascii=False) if isinstance(function_arguments, dict) else function_arguments,
                                              "name": function_name},
                                                       "type": 'function', "index": 0}]))
                self.dialogue.put(Message(role="tool", tool_call_id=function_id, content=result.response))
                self.dialogue.put(Message(**result.result))
                self.dialogue.put(Message(role="user", content="ok"))
                sub_response = self.chat_tool(query, depth=depth + 1, session_id=session_id)
                if sub_response:
                    response_message.extend(sub_response)
            else:
                logger.error(f"未知的 Action 类型: {result.action}")
        
        return response_message

    def chat(self, query):
        if query:
            logger.info(f"开始处理对话: {query}")
            self.dialogue.put(Message(role="user", content=query))
        else:
            logger.info("继续之前的对话处理")
            
        response_message = []
        start = 0
        self.chat_lock = True
        current_session = self.chat_session_id
        try:
            if self.start_task_mode:
                logger.debug("进入任务模式 (chat_tool)")
                # 获取当前对话历史，让 chat_tool 自己处理
                response_message_tool = self.chat_tool(query, session_id=current_session)
                if isinstance(response_message_tool, list):
                    response_message.extend(response_message_tool)
                else:
                    # 如果 chat_tool 递归调用了 chat，它会返回 True/None
                    return response_message_tool
            else:
                # 提交 LLM 任务
                logger.debug("进入普通对话模式 (llm.response)")
                try:
                    start_time = time.time()  # 记录开始时间
                    llm_responses = self.llm.response(self.dialogue.get_llm_dialogue(max_history=self.max_history))
                except Exception as e:
                    self.chat_lock = False
                    logger.error(f"LLM 处理出错 {query}: {e}", exc_info=True)
                    return None
                
                # 提交 TTS 任务到线程池
                logger.debug("开始迭代 LLM 响应")
                for content in llm_responses:
                    if self.stop_event.is_set():
                        logger.info("检测到 stop_event，停止 LLM 响应迭代")
                        break
                    
                    if not self.chat_lock or current_session != self.chat_session_id:
                        logger.info(f"检测到会话失效 (lock={self.chat_lock}, session={current_session}/{self.chat_session_id})，停止 LLM 响应迭代")
                        break
                    
                    response_message.append(content)
                    response_message_concat = "".join(response_message)
                    
                    # 过滤掉思考部分再进行分句
                    clean_response_concat = remove_think_tags(response_message_concat)
                    
                    # 循环查找所有可能的分句，处理一个 chunk 中包含多个句子的情况
                    while True:
                        flag_segment, index_segment = is_segment_sentence(clean_response_concat, start)
                        if flag_segment:
                            segment_text = clean_response_concat[start:index_segment + 1].strip()
                            if len(segment_text) >= 1:
                                logger.info(f"[{current_session}] 识别到分句 (start={start}, end={index_segment}): {segment_text}")
                                future = self.executor.submit(self.speak_and_play, segment_text)
                                self.tts_queue.put((current_session, future))
                                logger.debug(f"[{current_session}] 分句已入队，当前队列大小: {self.tts_queue.qsize()}")
                            start = index_segment + 1
                        else:
                            break
                
                # 处理剩余的响应
                if self.chat_lock and current_session == self.chat_session_id:
                    clean_response_concat = remove_think_tags("".join(response_message))
                    if start < len(clean_response_concat):
                        segment_text = clean_response_concat[start:].strip()
                        if segment_text:
                            logger.info(f"[{current_session}] 识别到最后分句 (start={start}): {segment_text}")
                            future = self.executor.submit(self.speak_and_play, segment_text)
                            self.tts_queue.put((current_session, future))
                            logger.debug(f"[{current_session}] 最后分句已入队")
        finally:
            if hasattr(self.player, 'send_status'):
                self.player.send_status("idle")
            self.chat_lock = False
            logger.info(f"对话处理完成: {query}")

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
