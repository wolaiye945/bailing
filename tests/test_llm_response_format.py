import unittest
from unittest.mock import MagicMock, patch
from bailing.robot import Robot
import os

class TestLLMResponseFormat(unittest.TestCase):
    def setUp(self):
        # Mock the config and other dependencies to initialize Robot
        self.config_patcher = patch('bailing.robot.read_config')
        self.mock_read_config = self.config_patcher.start()
        self.mock_read_config.return_value = {
            "selected_module": {
                "Recorder": "MockRecorder",
                "ASR": "MockASR",
                "LLM": "MockLLM",
                "TTS": "MockTTS",
                "VAD": "MockVAD",
                "Player": "MockPlayer"
            },
            "Recorder": {"MockRecorder": {}},
            "ASR": {"MockASR": {}},
            "LLM": {"MockLLM": {}},
            "TTS": {"MockTTS": {}},
            "VAD": {"MockVAD": {}},
            "Player": {"MockPlayer": {}},
            "Memory": {"enabled": False, "dialogue_history_path": "tmp/"},
            "interrupt": True,
            "TaskManager": {"functions_call_name": "plugins/function_calls_config.json"},
            "StartTaskMode": False
        }

        # Mock the instance creation
        self.recorder_patcher = patch('bailing.recorder.create_instance')
        self.asr_patcher = patch('bailing.asr.create_instance')
        self.llm_patcher = patch('bailing.llm.create_instance')
        self.tts_patcher = patch('bailing.tts.create_instance')
        self.vad_patcher = patch('bailing.vad.create_instance')
        self.player_patcher = patch('bailing.player.create_instance')

        self.mock_recorder = self.recorder_patcher.start()
        self.mock_asr = self.asr_patcher.start()
        self.mock_llm = self.llm_patcher.start()
        self.mock_tts = self.tts_patcher.start()
        self.mock_vad = self.vad_patcher.start()
        self.mock_player = self.player_patcher.start()

        # Initialize Robot with mocked dependencies
        self.robot = Robot("dummy_config.yaml")
        # Mock speak_and_play to track calls
        self.robot.speak_and_play = MagicMock(return_value="mock_audio.wav")

    def tearDown(self):
        self.config_patcher.stop()
        self.recorder_patcher.stop()
        self.asr_patcher.stop()
        self.llm_patcher.stop()
        self.tts_patcher.stop()
        self.vad_patcher.stop()
        self.player_patcher.stop()

    def test_chat_filters_think_tags(self):
        # Simulate an LLM response stream with <think> tags
        mock_response_stream = [
            "<think>", "I should ", "greet the ", "user.", "</think>",
            "Hello! ", "How can I ", "help you ", "today?"
        ]
        self.robot.llm.response = MagicMock(return_value=mock_response_stream)
        
        # Run chat
        self.robot.chat("hi")
        
        # Check that speak_and_play was called with the non-thinking part
        # The segments might depend on is_segment_sentence logic
        # But certainly none of the calls should contain "<think>" or "I should greet"
        
        called_texts = [call.args[0] for call in self.robot.speak_and_play.call_args_list]
        full_voice_text = "".join(called_texts)
        
        self.assertNotIn("<think>", full_voice_text)
        self.assertNotIn("I should greet", full_voice_text)
        self.assertIn("Hello!", full_voice_text)
        self.assertIn("How can I help you today?", full_voice_text)

    def test_chat_no_think_tags(self):
        # Simulate a normal LLM response
        mock_response_stream = ["Hello! ", "How are you?"]
        self.robot.llm.response = MagicMock(return_value=mock_response_stream)
        
        self.robot.chat("hi")
        
        called_texts = [call.args[0] for call in self.robot.speak_and_play.call_args_list]
        full_voice_text = "".join(called_texts)
        
        self.assertEqual(full_voice_text, "Hello! How are you?")

if __name__ == '__main__':
    unittest.main(exit=False)
    os._exit(0)
