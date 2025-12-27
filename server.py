import threading
import time

from fastapi import FastAPI, WebSocket, Query, WebSocketDisconnect, Form, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, FileResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware
import uvicorn
from contextlib import asynccontextmanager
import asyncio

import os
import argparse
import json
import logging
import socket
import shutil
import re
from typing import Dict, Tuple, List


parser = argparse.ArgumentParser(description="Description of your script.")

# Add arguments
parser.add_argument('--config_path', type=str, help="配置文件", default="config/config.yaml")
parser.add_argument('--debug', action='store_true', help="开启调试模式 (热重载)")

# Parse arguments
args = parser.parse_args()
config_path = args.config_path
debug_mode = args.debug

# 配置日志记录
log_level = logging.DEBUG if debug_mode else logging.INFO
logging.basicConfig(
    level=log_level,
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

# 读取配置文件
config = read_config(config_path)
server_config = config.get("Server", {})
host = server_config.get("host", "0.0.0.0")
port = server_config.get("port", 8000)

# 安全配置
security_config = config.get("Security", {})
SECURITY_ENABLED = security_config.get("enabled", False)
# 将用户列表转换为字典，方便查找
USERS_CONFIG = {u["username"]: u for u in security_config.get("users", [])}
# 保留旧配置的兼容性，如果没有 users 列表则使用旧的单用户配置
if not USERS_CONFIG:
    auth_username = security_config.get("username", "admin")
    auth_password = security_config.get("password", "bailing123")
    USERS_CONFIG[auth_username] = {
        "username": auth_username,
        "password": auth_password,
        "role": "admin"
    }

SECRET_KEY = security_config.get("secret_key", "bailing_secret_key_change_me")


TIMEOUT = 600  # 600 秒不活跃断开
# active_robots: Dict[connection_id, [robot_instance, timestamp, user_info]]
active_robots: Dict[str, list] = {}

async def cleanup_task():
    while True:
        now = time.time()
        for conn_id, robot_data in list(active_robots.items()):
            robot_instance, ts, _ = robot_data
            if now - ts > TIMEOUT:
                try:
                    robot_instance.recorder.stop_recording()
                    robot_instance.shutdown()
                    logger.info(f"连接 {conn_id} 对应的robot因超时已释放")
                except Exception as e:
                    logger.info(f"连接 {conn_id} 对应的robot释放出错: {e}")
                active_robots.pop(conn_id, None)
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

# 身份验证辅助函数
async def get_current_user(request: Request):
    if not SECURITY_ENABLED:
        return {"username": "admin", "role": "admin"}
    user_data = request.session.get("user")
    if not user_data:
        return None
    return user_data

# 登录相关路由
@app.get("/login")
async def login_page():
    return FileResponse("static/login.html")

@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    user_config = USERS_CONFIG.get(username)
    if user_config and str(user_config["password"]) == str(password):
        user_data = {
            "username": username,
            "role": user_config.get("role", "user")
        }
        request.session["user"] = user_data
        # 兼容性：同时也存储 role
        request.session["role"] = user_data["role"]
        return JSONResponse({"status": "ok", "user": user_data})
    return JSONResponse({"status": "error", "message": "Invalid credentials"}, status_code=401)

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login")

# 中间件：检查身份验证
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if not SECURITY_ENABLED:
        return await call_next(request)
    
    # 允许访问登录页面和静态资源
    path = request.url.path
    if path in ["/login", "/logout"] or path.startswith("/static/"):
        return await call_next(request)
    
    # 检查会话
    user = request.session.get("user")
    if not user:
        # 如果是 API 请求，返回 401
        if path.startswith("/temp_files") or path.startswith("/admin/"):
            return JSONResponse({"status": "error", "message": "Unauthorized"}, status_code=401)
        # 如果是页面请求，重定向到登录页
        return RedirectResponse(url="/login")
    
    return await call_next(request)

# 添加 Session 中间件 (必须在 auth_middleware 之后添加，以确保它在 request 阶段先运行)
if SECURITY_ENABLED:
    app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)



@app.get("/api/me")
async def get_me(user: dict = Depends(get_current_user)):
    return user

@app.get("/temp_files")
async def get_temp_files(user: dict = Depends(get_current_user)):
    tmp_dir = "tmp"
    if not os.path.exists(tmp_dir):
        return {}
    
    is_admin = user.get("role") == "admin"
    current_username = user.get("username")
    
    result = {}

    def process_date_dir(d_path, d_name, user_prefix=""):
        if not os.path.isdir(d_path) or not re.match(r"\d{4}-\d{2}-\d{2}", d_name):
            return
        
        files = []
        for f in os.listdir(d_path):
            f_path = os.path.join(d_path, f)
            if os.path.isfile(f_path):
                # 记录相对路径，方便后续操作
                rel_path = os.path.join(user_prefix, d_name, f) if user_prefix else os.path.join(d_name, f)
                files.append({
                    "name": f,
                    "rel_path": rel_path.replace("\\", "/"),
                    "size": os.path.getsize(f_path),
                    "ctime": os.path.getctime(f_path),
                    "username": user_prefix or "default"
                })
        
        if files:
            if d_name not in result:
                result[d_name] = []
            result[d_name].extend(files)

    # 遍历 tmp 目录
    for entry in os.listdir(tmp_dir):
        entry_path = os.path.join(tmp_dir, entry)
        if not os.path.isdir(entry_path):
            continue
            
        if re.match(r"\d{4}-\d{2}-\d{2}", entry):
            # 旧结构: tmp/{date}/
            # 只有管理员可以看到旧结构的全部文件，普通用户看不了（因为没法确定归属）
            if is_admin:
                process_date_dir(entry_path, entry)
        else:
            # 新结构: tmp/{username}/{date}/
            username = entry
            if is_admin or username == current_username:
                for date_dir in os.listdir(entry_path):
                    date_path = os.path.join(entry_path, date_dir)
                    process_date_dir(date_path, date_dir, user_prefix=username)

    # 对每个日期的文件按时间排序
    for date_key in result:
        result[date_key] = sorted(result[date_key], key=lambda x: x["ctime"], reverse=True)
        
    # 按日期倒序排列
    return dict(sorted(result.items(), key=lambda x: x[0], reverse=True))

@app.delete("/temp_files")
async def delete_temp_files(files: List[str] = Query(...), user: dict = Depends(get_current_user)):
    tmp_dir = "tmp"
    deleted_count = 0
    is_admin = user.get("role") == "admin"
    current_username = user.get("username")

    for f_rel_path in files:
        # 安全检查：防止路径穿越
        if ".." in f_rel_path or f_rel_path.startswith("/") or f_rel_path.startswith("\\"):
            continue
            
        # 安全检查：非管理员只能删除自己的文件
        # 新结构下，相对路径应该是 {username}/{date}/file.wav
        parts = f_rel_path.replace("\\", "/").split("/")
        if len(parts) >= 3:
            file_username = parts[0]
            if not is_admin and file_username != current_username:
                continue
        elif not is_admin:
            # 旧结构文件 (date/file.wav)，非管理员禁止删除
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
async def delete_date_files(date_str: str, user: dict = Depends(get_current_user)):
    # 只有管理员可以删除整个日期目录
    if user.get("role") != "admin":
        return JSONResponse({"status": "error", "message": "Permission denied"}, status_code=403)

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

# 管理员专属 API
@app.get("/admin/active_connections")
async def get_active_connections(user: dict = Depends(get_current_user)):
    if user.get("role") != "admin":
        return JSONResponse({"status": "error", "message": "Permission denied"}, status_code=403)
    
    connections = []
    for conn_id, robot_data in active_robots.items():
        _, ts, user_info = robot_data
        connections.append({
            "connection_id": conn_id,
            "user": user_info,
            "active_since": ts,
            "idle_seconds": int(time.time() - ts)
        })
    return connections

@app.post("/admin/disconnect/{conn_id}")
async def disconnect_connection(conn_id: str, user: dict = Depends(get_current_user)):
    if user.get("role") != "admin":
        return JSONResponse({"status": "error", "message": "Permission denied"}, status_code=403)
    
    if conn_id in active_robots:
        try:
            robot_instance, _, _ = active_robots[conn_id]
            robot_instance.shutdown()
            active_robots.pop(conn_id, None)
            return {"status": "ok"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    return {"status": "not_found"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, user_id: str = Query(...), connection_id: str = Query(None)):
    # 检查身份验证
    if SECURITY_ENABLED:
        session_user = websocket.session.get("user")
        if not session_user:
            logger.warning(f"WebSocket 拒绝未授权访问: user_id={user_id}")
            await websocket.close(code=1008) # Policy Violation
            return
        
        # 确保 user_info 是字典且包含必要字段
        if isinstance(session_user, dict):
            user_info = session_user
        else:
            # 如果 session 中存的是字符串（用户名），则包装成字典
            user_info = {"username": str(session_user), "role": "user"}
    else:
        user_info = {"username": user_id, "role": "admin" if user_id == "admin" else "user"}
    
    # 如果没有提供 connection_id，生成一个
    if not connection_id:
        connection_id = f"{user_id}_{int(time.time()*1000)}"

    await websocket.accept()
    loop = asyncio.get_event_loop()
    logger.info(f"WebSocket连接已建立: user_id={user_id}, conn_id={connection_id}")
    
    # 如果已存在该连接的 robot，先关闭旧的 (通常在刷新页面时发生)
    if connection_id in active_robots:
        try:
            old_robot, _, _ = active_robots[connection_id]
            old_robot.shutdown()
            logger.info(f"清理旧的 robot 实例: conn_id={connection_id}")
        except Exception as e:
            logger.error(f"清理旧 robot 实例出错: {e}")
        active_robots.pop(connection_id, None)

    # 创建新的 robot 实例
    robot_instance = robot.Robot(config_path, websocket, loop, user_info)
    active_robots[connection_id] = [robot_instance, time.time(), user_info]
    logger.info(f"创建新 Robot 实例: user_id={user_id}, conn_id={connection_id}, robot_id={id(robot_instance)}")
    
    # 启动 robot 运行线程
    robot_thread = threading.Thread(target=robot_instance.run, daemon=True)
    robot_thread.start()

    try:
        while True:
            # 检查当前 robot 是否仍是该连接的活跃实例
            if active_robots.get(connection_id) and active_robots[connection_id][0] is not robot_instance:
                logger.info(f"检测到新的 Robot 实例已接管，退出旧连接循环: conn_id={connection_id}")
                break

            # 使用 receive() 并显式处理断开连接
            msg = await websocket.receive()
            
            if msg["type"] == "websocket.disconnect":
                logger.info(f"收到断开连接信号: conn_id={connection_id}")
                break

            if "bytes" in msg:
                # 记录收到的音频数据，但不频繁打印以避免刷屏
                # 可以在每收到 100 次二进制消息时打印一次日志
                if not hasattr(websocket, "_recv_count"):
                    websocket._recv_count = 0
                websocket._recv_count += 1
                if websocket._recv_count % 100 == 0:
                    logger.debug(f"WebSocket 收到二进制数据包: count={websocket._recv_count}, size={len(msg['bytes'])} bytes")
                
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
                    elif msg_js.get("type") == "playback_finished":
                        # 处理前端返回的播放完成信号
                        if hasattr(robot_instance.player, "_playback_finished_event"):
                            robot_instance.player._playback_finished_event.set()
                            logger.debug(f"收到前端播放完成确认: conn_id={connection_id}")
                    else:
                        logger.warning(f"未知指令类型: {msg_js.get('type')}")
                except json.JSONDecodeError:
                    logger.error(f"无法解析 JSON 文本消息: {msg['text']}")
            
            # 更新活跃时间
            if connection_id in active_robots:
                active_robots[connection_id][1] = time.time()

    except WebSocketDisconnect:
        logger.info(f"WebSocket 断开连接: conn_id={connection_id}")
    except Exception as e:
        logger.error(f"WebSocket 错误 (conn_id={connection_id}): {e}")
    finally:
        # 彻底清理资源
        try:
            if robot_instance:
                robot_instance.shutdown()
            # 只有当 active_robots 中的实例还是当前这个时才删除
            if active_robots.get(connection_id) and active_robots[connection_id][0] is robot_instance:
                active_robots.pop(connection_id, None)
            logger.info(f"资源已清理，WebSocket 连接关闭: conn_id={connection_id}")
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
        print("注意: 局域网访问或通过域名访问时，非 HTTPS 模式可能导致浏览器禁用麦克风。")
        print("如果您使用了反向代理（如 Nginx/Caddy），请确保代理配置了正确的 SSL 证书，并转发 X-Forwarded-Proto 头。")

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
        proxy_headers=True,
        forwarded_allow_ips="*",
        reload=debug_mode
    )