from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set
import time, math


@dataclass(frozen=True)
class Candidate:
    id: str
    text: str
    rel: float  # 相关性（dense/bm25/启发式）
    cov: Set[str]  # 覆盖要素（简单关键词/实体集合）
    cost_est: float  # 成本估计（token/时延）
    meta: Dict[str, Any]


@dataclass
class Budget:
    max_steps: Optional[int] = None
    max_tokens: Optional[int] = None
    wall_clock_s: Optional[float] = None
    spent_steps: int = 0
    spent_tokens: int = 0
    t0: float = time.time()

    def stop(self) -> bool:
        if self.wall_clock_s and time.time() - self.t0 >= self.wall_clock_s: return True
        if self.max_steps is not None and self.spent_steps >= self.max_steps: return True
        if self.max_tokens is not None and self.spent_tokens >= self.max_tokens: return True
        return False


class ValueAwareAStarSelector:
    """
    A* 风格的价值剪枝选择器：越小越好的 f(c) 打分
    f(c) = -(λ_rel*rel + λ_cov*Δcov - λ_red*redundancy) + λ_cost*cost
    """

    def __init__(self, *, top_k=8, tau=float("inf"),
                 lambda_rel=1.0, lambda_cov=0.5, lambda_red=0.1, lambda_cost=0.1):
        self.k = top_k
        self.tau = tau
        self.w = {"rel": lambda_rel, "cov": lambda_cov, "red": lambda_red, "cost": lambda_cost}
        self.seen_texts: Set[str] = set()
        self.used_cov: Set[str] = set()

    def _score(self, c: Candidate) -> float:
        cov_gain = len(c.cov - self.used_cov)
        red = 1.0 if (c.text in self.seen_texts) else 0.0
        return -(self.w["rel"] * c.rel + self.w["cov"] * cov_gain - self.w["red"] * red) \
            + self.w["cost"] * max(1.0, c.cost_est)

    def select(self, candidates: List[Candidate], budget: Budget) \
            -> Tuple[List[Candidate], Dict[str, Any], bool]:
        if not candidates:
            return [], {"best_f": 0.0, "num_in": 0, "num_kept": 0}, budget.stop()
        scored = sorted(((self._score(c), c) for c in candidates), key=lambda x: x[0])
        kept = [c for _, c in scored[:self.k]]
        best_f = scored[0][0]
        early = (best_f >= self.tau) or budget.stop()
        if kept:
            self.seen_texts.add(kept[0].text)
            self.used_cov |= kept[0].cov
        logs = {"best_f": best_f, "num_in": len(candidates), "num_kept": len(kept)}
        return kept, logs, early
