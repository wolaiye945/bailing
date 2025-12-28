import asyncio
import logging
import time
from bailing.utils import is_segment_sentence, remove_think_tags

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, session, asr, llm, tts, player, vad, nr, task_manager):
        self.session = session
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.player = player
        self.vad = vad
        self.nr = nr
        self.task_manager = task_manager
        
        self.audio_queue = asyncio.Queue()
        self.playback_queue = asyncio.Queue()
        self.stop_event = asyncio.Event()
        
    async def start(self):
        logger.info("Orchestrator starting...")
        tasks = [
            asyncio.create_task(self._process_audio_loop()),
            asyncio.create_task(self._tts_playback_loop())
        ]
        await asyncio.gather(*tasks)

    async def _process_audio_loop(self):
        """Handle VAD and ASR in an async loop"""
        speech_buffer = []
        vad_active = False
        
        while not self.stop_event.is_set():
            try:
                # This would come from the recorder (need to integrate with recorder.py)
                audio_chunk = await self.audio_queue.get()
                
                if self.nr.enabled:
                    audio_chunk = self.nr.process(audio_chunk)
                
                vad_status = self.vad.is_vad(audio_chunk)
                
                if "start" in vad_status:
                    logger.info("VAD started")
                    vad_active = True
                    speech_buffer = [audio_chunk]
                    # Handle interruption
                    if self.player.get_playing_status():
                        self.player.stop()
                        self.session.new_session() # Interrupt current session
                
                elif "end" in vad_status and vad_active:
                    logger.info("VAD ended")
                    vad_active = False
                    if speech_buffer:
                        # Start ASR and Chat in background
                        asyncio.create_task(self._handle_asr_and_chat(speech_buffer))
                    speech_buffer = []
                
                elif vad_active:
                    speech_buffer.append(audio_chunk)
                    
            except Exception as e:
                logger.error(f"Error in audio loop: {e}")
                await asyncio.sleep(0.1)

    async def _handle_asr_and_chat(self, audio_data):
        try:
            # ASR (using the new async_recognizer)
            text, _ = await self.asr.async_recognizer(audio_data, self.session.user_id)
            
            if not text or not text.strip():
                return
                
            logger.info(f"ASR Result: {text}")
            self.session.add_message("user", text)
            
            # LLM + TTS Pipeline
            await self._chat_pipeline(text)
            
        except Exception as e:
            logger.error(f"Error handling ASR/Chat: {e}")

    async def _chat_pipeline(self, query):
        session_id = self.session.session_id
        self.session.set_lock(True)
        
        try:
            # LLM Response (Streaming)
            messages = self.session.get_dialogue_history()
            
            full_response = ""
            current_segment = ""
            
            # Using async_response_call for streaming
            async for content, tool_calls in self.llm.async_response_call(messages, self.task_manager.get_tools()):
                if self.session.session_id != session_id:
                    logger.info("Session changed, interrupting LLM response")
                    break
                
                if content:
                    full_response += content
                    current_segment += content
                    
                    # Check if we have a complete sentence for TTS
                    if is_segment_sentence(current_segment):
                        clean_segment = remove_think_tags(current_segment).strip()
                        if clean_segment:
                            # Start TTS in background and queue for playback
                            asyncio.create_task(self._process_tts_and_queue(clean_segment, session_id))
                        current_segment = ""
                
                if tool_calls:
                    # Handle tool calls (Function Calling)
                    logger.info(f"Tool calls detected: {tool_calls}")
                    # Tool execution logic would go here
            
            # Handle remaining segment
            if current_segment and self.session.session_id == session_id:
                clean_segment = remove_think_tags(current_segment).strip()
                if clean_segment:
                    await self._process_tts_and_queue(clean_segment, session_id)
            
            if full_response:
                self.session.add_message("assistant", full_response)
                
        except Exception as e:
            logger.error(f"Error in chat pipeline: {e}")
        finally:
            self.session.set_lock(False)

    async def _process_tts_and_queue(self, text, session_id):
        """Convert text to speech and add to playback queue"""
        try:
            if not text:
                return
                
            # Use memory stream optimization
            audio_data = await self.tts.to_tts_stream(text)
            
            if audio_data and self.session.session_id == session_id:
                await self.playback_queue.put((audio_data, session_id))
                
        except Exception as e:
            logger.error(f"Error in TTS processing: {e}")

    async def _tts_playback_loop(self):
        """Handle audio playback in an async loop"""
        while not self.stop_event.is_set():
            try:
                audio_data, session_id = await self.playback_queue.get()
                
                # Only play if session matches (hasn't been interrupted)
                if session_id == self.session.session_id:
                    # player.play is typically blocking, run in executor
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self.player.play, audio_data)
                
                self.playback_queue.task_done()
            except Exception as e:
                logger.error(f"Error in playback loop: {e}")
                await asyncio.sleep(0.1)

    def shutdown(self):
        self.stop_event.set()
