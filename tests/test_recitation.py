import sys
import os
import time
import logging
import yaml

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bailing.robot import Robot

def test_recitation():
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        logger.error(f"配置文件不存在: {config_path}")
        return

    # 加载并修改配置以适应本地测试
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 强制使用本地播放器和录音器，避免 WebSocket 报错
    config["selected_module"]["Player"] = "PygamePlayer"
    config["selected_module"]["Recorder"] = "RecorderPyAudio"
    
    logger.info("正在初始化 Robot (这可能需要一些时间加载模型)...")
    temp_config_path = "config/config_test.yaml"
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
    
    robot = None
    try:
        logger.info("开始创建 Robot 实例...")
        robot = Robot(temp_config_path)
        logger.info("Robot 实例创建成功！")
        
        # 模拟用户输入
        query = "请朗诵一下诸葛亮的诫子书全文"
        logger.info(f"发送指令: {query}")
        
        # 执行对话
        robot.chat(query)
        
        logger.info("正在等待 LLM 和 TTS 处理（请听音频顺序）...")
        
        # 设置最大等待时间（3分钟）
        max_wait = 180
        start_wait = time.time()
        
        while time.time() - start_wait < max_wait:
            is_llm_generating = robot.chat_lock
            is_tts_queue_empty = robot.tts_queue.empty()
            is_playing = robot.player.get_playing_status()
            
            logger.info(f"等待中... [LLM正在生成: {is_llm_generating}, TTS队列空: {is_tts_queue_empty}, 正在播放: {is_playing}]")
            
            if not is_llm_generating and is_tts_queue_empty and not is_playing:
                logger.info("所有音频播放完毕。")
                break
            time.sleep(2)
        else:
            logger.warning("达到最大等待时间，强制结束测试。")
            
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
    except Exception as e:
        logger.error(f"测试过程中出错: {e}", exc_info=True)
    finally:
        if robot:
            robot.shutdown()
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        logger.info("测试结束")

if __name__ == "__main__":
    test_recitation()
