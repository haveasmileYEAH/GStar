# scripts/memory_router_demo.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # 保证能 import 根目录模块
from gstar_astar_proto import answer_with_memory_router_bp

if __name__ == "__main__":
    q1 = "How to put the apple into the fridge?"
    ans1, meta1 = answer_with_memory_router_bp(q1)       # 第一次：novel → 走 LLM，写入记忆
    print("Q1 route:", meta1)

    q2 = "How to put an apple into the refrigerator?"
    ans2, meta2 = answer_with_memory_router_bp(q2)       # 相近：partial → 从断点追加到同一条记忆
    print("Q2 route:", meta2)
