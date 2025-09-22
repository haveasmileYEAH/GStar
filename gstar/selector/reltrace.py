# gstar/selector/reltrace.py
from collections import deque


class RelTrace:
    def __init__(self, k: int = 3):
        self.k = max(2, k)
        self.q = deque(maxlen=self.k)

    def push(self, x: float):
        self.q.append(float(x))

    def cliff(self, tau_drop: float) -> tuple[bool, float]:
        """返回(是否崖降, Δ值)，使用最近两次差分；也可以换成均值/EMWA"""
        if len(self.q) < 2:
            return (False, 0.0)
        delta = self.q[-1] - self.q[-2]
        return (delta <= -abs(tau_drop), delta)
