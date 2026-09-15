import os
from pathlib import Path
# 设置模型下载镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'      # 强制离线
os.environ['TRANSFORMERS_OFFLINE'] = '1'
# 缓存设为「脚本所在目录」下的 hf_cache
SCRIPT_DIR = Path(__file__).resolve().parent
os.environ['HF_HOME'] = str(SCRIPT_DIR / 'hf_cache')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 指定模型名称
model_name = "distilgpt2"

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() 
                      else "cpu")
model.to(device)

# 设置模型为评估模式
model.eval()

# 输入文本
input_text = "Hello GPT"

# 编码输入文本
inputs = tokenizer(input_text, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}

# 生成文本
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=200,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

# 解码生成的文本
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("模型生成的文本：")
print(generated_text)