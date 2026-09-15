from openai import OpenAI
import os
from dotenv import load_dotenv
import gradio as gr
import json

load_dotenv()

# 初始化 OpenAI 客户端，使用阿里云 DashScope API
client = OpenAI(
    api_key=os.getenv('API_KEY'), # 如果你没有配置环境变量，使用 api_key="your-api-key" 替换
    base_url=os.getenv('OPENAI_URL'), # 这里使用的是阿里云的大模型，如果需要使用其他平台，请参考对应的开发文档后对应修改
)

# 检查 API 设置是否正确
try:
    response = client.chat.completions.create(
        model="qwen-turbo",  # 使用通义千问-Turbo 大模型，可以替换为 Deepseek 系列：deepseek-v3 / deepseek-r1
        messages=[{'role': 'user', 'content': "测试"}],
        max_tokens=1,
    )
    print("API 设置成功！！")
except Exception as e:
    print(f"API 可能有问题，请检查：{e}")

# TODO: 修改以下变量以定义角色和角色提示词
CHARACTER_FOR_CHATBOT = "猫娘"  # 机器人扮演的角色，注意，真正起作用的实际是提示词，因为并没有预设 system 角色
PROMPT_FOR_ROLEPLAY = "我需要你以猫娘女仆的身份进行聊天"  # 指定角色提示词

# 清除对话的函数
def reset():
    """
    清空对话记录。

    返回:
        List: 空的对话记录列表
    """
    return []

# 调用模型生成对话的函数
def interact_roleplay(chatbot, user_input, temp=1.0):
    """
    处理角色扮演多轮对话，调用模型生成回复。

    参数:
        chatbot (List[Tuple[str, str]]): 对话历史记录（用户与模型回复）
        user_input (str): 当前用户输入
        temp (float): 模型温度参数（默认 1.0）

    返回:
        List[Tuple[str, str]]: 更新后的对话记录
    """
    try:
        # 直接复制历史（history 已经是 messages 格式）
        messages = list(chatbot)
        # 再添加本轮用户输入
        messages.append({'role': 'user', 'content': user_input})
        
        # 调用 API 获取回复
        response = client.chat.completions.create(
            model="qwen-turbo",
            messages=messages,
            temperature=temp,
        )
        chatbot.append({'role': 'user', 'content': user_input})
        chatbot.append({'role': 'assistant', 'content': response.choices[0].message.content})

    except Exception as e:
        print(f"发生错误：{e}")
        chatbot.append((user_input, f"抱歉，发生了错误：{e}"))
        
    return chatbot

def export_roleplay(chatbot, description):
    """
    导出角色扮演对话记录及任务描述到 JSON 文件。

    参数:
        chatbot (List[Tuple[str, str]]): 对话记录
        description (str): 任务描述
    """
    target = {"chatbot": chatbot, "description": description}
    with open("files/part2.json", "w", encoding="utf-8") as file:
        json.dump(target, file, ensure_ascii=False, indent=4)

# 进行第一次对话：设定角色提示
first_dialogue = [
    {"role": "system", "content": PROMPT_FOR_ROLEPLAY},
]

# 构建 Gradio UI 界面
with gr.Blocks() as demo:
    gr.Markdown("# 第2部分：角色扮演\n与聊天机器人进行角色扮演互动！")
    chatbot = gr.Chatbot(value=first_dialogue)
    description_textbox = gr.Textbox(label="机器人扮演的角色", interactive=False, value=CHARACTER_FOR_CHATBOT)
    input_textbox = gr.Textbox(label="输入", value="")
    
    with gr.Column():
        gr.Markdown("# 温度调节\n温度控制聊天机器人的响应创造性。")
        temperature_slider = gr.Slider(0.0, 1.99, 1.0, step=0.01, label="温度")
    
    with gr.Row():
        send_button = gr.Button(value="发送")
        reset_button = gr.Button(value="重置")
    
    with gr.Column():
        gr.Markdown("# 保存结果\n点击导出按钮保存对话记录。")
        export_button = gr.Button(value="导出")
        
    # 绑定按钮与回调函数
    send_button.click(interact_roleplay, inputs=[chatbot, input_textbox, temperature_slider], outputs=[chatbot])
    reset_button.click(reset, outputs=[chatbot])
    export_button.click(export_roleplay, inputs=[chatbot, description_textbox])
    
# 启动 Gradio 应用
demo.launch(debug=True)