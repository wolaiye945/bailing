import sys
import os
import pytest
from fastapi.testclient import TestClient

# 将项目根目录添加到 python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app, AUTH_USERNAME, AUTH_PASSWORD, SECURITY_ENABLED

client = TestClient(app)

def test_login_flow():
    if not SECURITY_ENABLED:
        pytest.skip("Security is disabled in config")

    # 1. 访问首页，应该被重定向到 /login
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/login"

    # 2. 尝试错误密码登录
    response = client.post("/login", data={"username": "admin", "password": "wrong_password"})
    assert response.status_code == 401
    assert response.json()["status"] == "error"

    # 3. 使用正确密码登录
    response = client.post("/login", data={"username": AUTH_USERNAME, "password": AUTH_PASSWORD})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    
    # 获取 session cookie
    session_cookie = response.cookies.get("session")
    assert session_cookie is not None

    # 4. 登录后访问首页
    response = client.get("/", cookies={"session": session_cookie}, follow_redirects=False)
    assert response.status_code == 200

    # 5. 测试注销
    response = client.get("/logout", cookies={"session": session_cookie}, follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/login"

if __name__ == "__main__":
    # 如果没有安装 pytest，直接运行这个函数
    try:
        test_login_flow()
        print("Auth logic test passed!")
    except Exception as e:
        print(f"Auth logic test failed: {e}")
        sys.exit(1)
