from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv(override=True)
api_key = (os.getenv('DASHSCOPE_API_KEY') or '').strip()
base_url = os.getenv('QWEN_BASE_URL','https://dashscope.aliyuncs.com/compatible-mode/v1')
model    = os.getenv('QWEN_MODEL','qwen-plus')  # 先用通用型号试

assert api_key, 'DASHSCOPE_API_KEY is empty'

client = OpenAI(api_key=api_key, base_url=base_url)
r = client.chat.completions.create(
    model=model,
    messages=[{'role':'user','content':'只回复：pong'}],
    temperature=0
)
print(r.choices[0].message.content)
