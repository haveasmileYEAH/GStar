from dotenv import load_dotenv
import os
load_dotenv(override=True)
k = (os.getenv('DASHSCOPE_API_KEY') or '').strip()
print('KEY exists:', bool(k), 'len:', len(k) if k else 0)
print('BASE_URL  :', os.getenv('QWEN_BASE_URL'))
print('MODEL     :', os.getenv('QWEN_MODEL'))
