import threading
import time

from fastapi import FastAPI, WebSocket, Query, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
import asyncio

import argparse
import json
import logging
import socket
import shutil
import re
from typing import Dict, Tuple, List


# 配置日志记录
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 控制台输出
        logging.FileHandler('tmp/bailing.log')  # 文件输出
    ]
)
from bailing import robot
from bailing.utils import read_config

# 获取根 logger
logger = logging.getLogger(__name__)



parser = argparse.ArgumentParser(description="Description of your script.")

# Add arguments
parser.add_argument('--config_path', type=str, help="配置文件", default="config/config.yaml")
parser.add_argument('--debug', action='store_true', help="开启调试模式 (热重载)")

# Parse arguments
args = parser.parse_args()
config_path = args.config_path
debug_mode = args.debug

# 读取配置文件
config = read_config(config_path)
server_config = config.get("Server", {})
host = server_config.get("host", "0.0.0.0")
port = server_config.get("port", 8000)


app = FastAPI()
TIMEOUT = 600  # 60 秒不活跃断开
active_robots: Dict[str, list] = {}

async def cleanup_task():
    while True:
        now = time.time()
        for uid, (robot_instance, ts) in list(active_robots.items()):
            if now - ts > TIMEOUT:
                try:
                    robot_instance.recorder.stop_recording()
                    robot_instance.shutdown()
                    logger.info(f"{uid} 对应的robot已释放")
                except Exception as e:
                    logger.info(f"{uid} 对应的robot释放 出错: {e}")
                active_robots.pop(uid, None)
        await asyncio.sleep(10)

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(cleanup_task())
    yield
    task.cancel()
    await task

app = FastAPI(lifespan=lifespan)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/temp_files")
async def get_temp_files():
    tmp_dir = "tmp"
    if not os.path.exists(tmp_dir):
        return {}
    
    result = {}
    for date_dir in sorted(os.listdir(tmp_dir), reverse=True):
        date_path = os.path.join(tmp_dir, date_dir)
        if os.path.isdir(date_path) and re.match(r"\d{4}-\d{2}-\d{2}", date_dir):
            files = []
            for f in os.listdir(date_path):
                f_path = os.path.join(date_path, f)
                if os.path.isfile(f_path):
                    files.append({
                        "name": f,
                        "size": os.path.getsize(f_path),
                        "ctime": os.path.getctime(f_path)
                    })
            if files:
                result[date_dir] = sorted(files, key=lambda x: x["ctime"], reverse=True)
    return result

@app.delete("/temp_files")
async def delete_temp_files(files: List[str] = Query(...)):
    tmp_dir = "tmp"
    deleted_count = 0
    for f_rel_path in files:
        # 简单安全检查，防止路径穿越
        if ".." in f_rel_path or f_rel_path.startswith("/") or f_rel_path.startswith("\\"):
            continue
            
        full_path = os.path.join(tmp_dir, f_rel_path)
        if os.path.exists(full_path) and os.path.isfile(full_path):
            try:
                os.remove(full_path)
                deleted_count += 1
            except Exception as e:
                logger.error(f"删除文件失败 {full_path}: {e}")
                
    return {"status": "ok", "deleted_count": deleted_count}

@app.delete("/temp_files/date/{date_str}")
async def delete_date_files(date_str: str):
    tmp_dir = "tmp"
    date_path = os.path.join(tmp_dir, date_str)
    if os.path.exists(date_path) and os.path.isdir(date_path):
        try:
            shutil.rmtree(date_path)
            return {"status": "ok"}
        except Exception as e:
            logger.error(f"删除目录失败 {date_path}: {e}")
            return {"status": "error", "message": str(e)}
    return {"status": "not_found"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, user_id: str = Query(...)):
    await websocket.accept()
    loop = asyncio.get_event_loop()
    logger.info(f"WebSocket连接已建立: user_id={user_id}")
    
    # 如果已存在该用户的 robot，先关闭旧的
    if user_id in active_robots:
        try:
            old_robot, _ = active_robots[user_id]
            old_robot.shutdown()
            logger.info(f"清理旧的 robot 实例: user_id={user_id}")
        except Exception as e:
            logger.error(f"清理旧 robot 实例出错: {e}")
        active_robots.pop(user_id, None)

    # 创建新的 robot 实例
    robot_instance = robot.Robot(config_path, websocket, loop)
    active_robots[user_id] = [robot_instance, time.time()]
    logger.info(f"创建新 Robot 实例: user_id={user_id}, robot_id={id(robot_instance)}")
    
    # 启动 robot 运行线程
    robot_thread = threading.Thread(target=robot_instance.run, daemon=True)
    robot_thread.start()

    try:
        while True:
            # 检查当前 robot 是否仍是该用户的活跃实例
            if active_robots.get(user_id) and active_robots[user_id][0] is not robot_instance:
                logger.info(f"检测到新的 Robot 实例已接管，退出旧连接循环: user_id={user_id}, old_robot_id={id(robot_instance)}")
                break

            # 使用 receive() 并显式处理断开连接
            msg = await websocket.receive()
            
            if msg["type"] == "websocket.disconnect":
                logger.info(f"收到断开连接信号: user_id={user_id}")
                break

            if "bytes" in msg:
                robot_instance.recorder.put_audio(msg["bytes"])
            elif "text" in msg:
                logger.info(f"收到请求: {msg['text']}")
                try:
                    msg_js = json.loads(msg["text"])
                    if msg_js.get("type") == "playback_status":
                        if msg_js.get("status") == "playing" or msg_js.get("queue_size", 0) > 0:
                            robot_instance.player.set_playing_status(True)
                        else:
                            robot_instance.player.set_playing_status(False)
                    else:
                        logger.warning(f"未知指令类型: {msg_js.get('type')}")
                except json.JSONDecodeError:
                    logger.error(f"无法解析 JSON 文本消息: {msg['text']}")
            
            # 更新活跃时间
            active_robots[user_id][1] = time.time()

    except WebSocketDisconnect:
        logger.info(f"WebSocket 断开连接: user_id={user_id}")
    except Exception as e:
        logger.error(f"WebSocket 错误 (user_id={user_id}): {e}")
    finally:
        # 彻底清理资源
        try:
            if robot_instance:
                robot_instance.shutdown()
            # 只有当 active_robots 中的实例还是当前这个时才删除
            if active_robots.get(user_id) and active_robots[user_id][0] is robot_instance:
                active_robots.pop(user_id, None)
            logger.info(f"资源已清理，WebSocket 连接关闭: user_id={user_id}")
        except Exception as e:
            logger.error(f"清理资源时出错: {e}")

# 托管前端静态文件
app.mount("/", StaticFiles(directory="static", html=True), name="static")

def get_lan_ip():
    try:
        # 创建一个UDP套接字
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # 连接到Google DNS
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        return "无法获取IP: " + str(e)


if __name__ == "__main__":
    lan_ip = get_lan_ip()
    
    # 如需在局域网使用麦克风，请在 ssl 目录下放置 key.pem 和 cert.pem，或使用以下命令生成：
    # openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes
    import os
    ssl_keyfile = "ssl/key.pem"
    ssl_certfile = "ssl/cert.pem"
    protocol = "https"
    
    if not os.path.exists(ssl_keyfile) or not os.path.exists(ssl_certfile):
        ssl_keyfile = None
        ssl_certfile = None
        protocol = "http"
        print("Warning: SSL 证书未找到 (ssl/key.pem, ssl/cert.pem)，将以 HTTP 模式启动。")
        print("注意: 局域网访问时，非 HTTPS 模式可能导致浏览器禁用麦克风。")

    print(f"\n请在局域网中使用以下地址访问:")
    print(f"{protocol}://{lan_ip}:{port}\n")

    uvicorn.run(
        "server:app" if debug_mode else app,
        host=host,
        port=port,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        ws_ping_interval=20,
        ws_ping_timeout=30,
        reload=debug_mode
    )