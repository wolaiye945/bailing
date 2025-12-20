import yaml
import json
import re


def load_prompt(prompt_path):
    with open(prompt_path, "r", encoding="utf-8") as file:
        prompt = file.read()
    return prompt.strip()


def read_json_file(file_path):
    """读取 JSON 文件并返回内容"""
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
            return data
        except json.JSONDecodeError as e:
            print(f"解析 JSON 时出错: {e}")
            return None

def write_json_file(file_path, data):
    """将数据写入 JSON 文件"""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def read_config(config_path):
    with open(config_path, "r",encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def is_segment(tokens):
    if tokens[-1] in (",", ".", "?", "，", "。", "？", "！", "!", ";", "；", ":", "："):
        return True
    else:
        return False

def is_segment_sentence(tokens, start_index):
    for i in range(len(tokens) - 1, start_index - 1, -1):
        if tokens[i] in (",", ".", "?", "，", "。", "？", "！", "!", ";", "；", ":", "："):
            return True, i
    return False, None

def is_interrupt(query: str):
    for interrupt_word in ("停一下", "听我说", "不要说了", "stop", "hold on", "excuse me"):
        if query.lower().find(interrupt_word)>=0:
            return True
    return False

def extract_json_from_string(input_string):
    """提取字符串中的 JSON 部分"""
    pattern = r'(\{.*\})'
    match = re.search(pattern, input_string)
    if match:
        return match.group(1)  # 返回提取 of the extracted JSON string
    return None

def remove_think_tags(text):
    """移除 <think>...</think> 标签及其内容，包括未闭合的标签内容"""
    if text is None:
        return ""
    # 先移除所有已闭合的标签
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 再移除可能存在的未闭合标签（从 <think> 开始到最后）
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    return text.strip()

def format_think_sections(text):
    """将 <think> 标签转换为更易读的格式（例如 Markdown 引用块）"""
    if text is None:
        return ""
    
    def replace_think(match):
        content = match.group(1).strip()
        if not content:
            return ""
        # 将内容每一行都加上引用符号
        quoted_content = "\n".join([f"> {line}" for line in content.split("\n")])
        return f"\n\n> **思考过程**\n{quoted_content}\n\n"

    # 处理闭合标签
    text = re.sub(r'<think>(.*?)</think>', replace_think, text, flags=re.DOTALL)
    
    # 处理未闭合标签（如果是流式输出中）
    if '<think>' in text:
        parts = text.split('<think>', 1)
        before = parts[0]
        after = parts[1]
        quoted_after = "\n".join([f"> {line}" for line in after.split("\n")])
        text = f"{before}\n\n> **思考过程 (正在思考...)**\n{quoted_after}"
        
    return text.strip()
