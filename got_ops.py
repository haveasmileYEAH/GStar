# got_ops.py —— Value-aware + improved auto-rho + gentle edge q-penalty
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Iterable, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class EdgeCand:
    j: int
    gen_tag: str
    score: float = 0.0
    cost: float = 0.0
    value: float = 0.0

class TFIDFGoTPolicy:
    def __init__(
        self,
        doc_mat,
        sim_q: np.ndarray,
        k_doc=8, kq=3, q_lambda=0.55,
        sim_mode="pct", sim_th=0.30, sim_pct=0.80,
        keep_n=6, ensure_deg=2,
        select_mode: str = "value",
        token_costs: Optional[np.ndarray] = None,
        rho: float = 0.35,
        lam_cost: float = 0.10,
        budget_tokens: Optional[int] = None,
        auto_rho: bool = True, rho_min: float = 0.30, rho_max: float = 0.70,
        edge_gamma_q: float = 0.05,       # 更温和
        edge_q_floor: float = 0.10        # 查询惩罚的相似度下限：sim_q[j] = max(sim_q[j], floor)
    ):
        self.doc_mat = doc_mat
        self.dd = cosine_similarity(doc_mat).astype(np.float32)
        self.sim_q = sim_q.astype(np.float32)
        self.k_doc, self.kq = k_doc, kq
        self.q_lambda = q_lambda
        self.sim_mode, self.sim_th, self.sim_pct = sim_mode, sim_th, sim_pct
        self.keep_n, self.ensure_deg = keep_n, ensure_deg
        self.select_mode = select_mode
        self.rho = rho
        self.lam_cost = lam_cost
        self.budget_tokens = budget_tokens
        self.auto_rho = auto_rho
        self.rho_min, self.rho_max = rho_min, rho_max
        self.edge_gamma_q = edge_gamma_q
        self.edge_q_floor = edge_q_floor

        if token_costs is None:
            self.token_costs = np.ones(self.dd.shape[0], dtype=np.float32)
        else:
            tc = np.asarray(token_costs, dtype=np.float32)
            tc[tc <= 0] = 1.0
            self.token_costs = tc
        self.token_norm = float(np.mean(self.token_costs))

    # ----- Generate -----
    def _row_th(self, i:int):
        row = self.dd[i].copy(); row[i] = -1.0
        if self.sim_mode == "pct" and np.any(row >= 0):
            return max(self.sim_th, float(np.quantile(row[row >= 0], self.sim_pct)))
        return self.sim_th

    def generate(self, i:int) -> Iterable[EdgeCand]:
        N = self.dd.shape[0]
        th = self._row_th(i)
        kk = max(1, min(self.k_doc, N-1))
        # doc-KNN
        idx = np.argpartition(-self.dd[i], kk)[:kk+1]
        idx = idx[np.argsort(-self.dd[i][idx])]
        for j in idx:
            if j != i and self.dd[i, j] >= th:
                yield EdgeCand(j=j, gen_tag="doc_knn")
        # query-aware
        pair_q = np.sqrt(np.outer(self.sim_q, self.sim_q))[i]
        s_all = (1.0 - self.q_lambda) * self.dd[i] + self.q_lambda * pair_q
        order = np.argsort(-s_all)
        added = set()
        for j in order:
            if j == i or j in added:
                continue
            yield EdgeCand(j=j, gen_tag="qaware")
            added.add(j)
            if len(added) >= max(self.k_doc, self.kq):
                break

    # ----- Score -----
    def score(self, i:int, cands:List[EdgeCand]) -> None:
        pair_q = np.sqrt(np.outer(self.sim_q, self.sim_q))[i]
        for c in cands:
            s = (1.0 - self.q_lambda) * float(self.dd[i, c.j]) + self.q_lambda * float(pair_q[c.j])
            c.score = max(0.0, min(1.0, s))
            c.cost  = 1.0 - c.score

    # ----- improved auto-ρ：用本行 doc 相似度的 90 分位数决定 ρ -----
    def _auto_rho_for_node(self, i:int, pool:List[EdgeCand]) -> float:
        if not self.auto_rho or not pool:
            return self.rho
        row = self.dd[i].copy(); row[i] = -1.0
        pos = row[row >= 0.0]
        if pos.size == 0:
            return self.rho
        p90 = float(np.quantile(pos, 0.90))
        # 将 p90 从区间 [0.05, 0.30] 线性映射到 [rho_min, rho_max]
        lo, hi = 0.05, 0.30
        t = (p90 - lo) / (hi - lo + 1e-9)
        t = max(0.0, min(1.0, t))
        return float(self.rho_min + (self.rho_max - self.rho_min) * t)

    # ----- Select -----
    def select(self, i:int, cands:List[EdgeCand]) -> List[EdgeCand]:
        # 去重
        uniq = {}
        for c in cands:
            if (c.j not in uniq) or (c.score > uniq[c.j].score):
                uniq[c.j] = c
        pool = list(uniq.values())

        if self.select_mode == "topk":
            picked = sorted(pool, key=lambda x: -x.score)[:max(self.keep_n, self.ensure_deg)]
            for c in picked:
                c.value = c.score
            return picked

        # value 模式：源多样性 + 温和查询惩罚（带 floor）
        selected: List[EdgeCand] = []
        remain = pool.copy()
        spent_tokens = 0
        rho_i = self._auto_rho_for_node(i, pool)

        def redundancy(j:int) -> float:
            red_src = float(self.dd[i, j])
            if not selected:
                return red_src
            sel_idx = np.array([c.j for c in selected], dtype=int)
            red_sel = float(np.max(self.dd[j, sel_idx]))
            return max(red_src, red_sel)

        while remain and len(selected) < max(self.keep_n, self.ensure_deg):
            best, best_key = None, -1e9
            for c in remain:
                red  = redundancy(c.j)
                sq   = max(float(self.sim_q[c.j]), self.edge_q_floor)  # floor
                qmis = 1.0 - sq
                token = float(self.token_costs[c.j])
                val = (1.0 - rho_i) * c.score + rho_i * (1.0 - red) \
                      - self.lam_cost * (token / self.token_norm) \
                      - self.edge_gamma_q * qmis
                dens = val / (token + 1e-6)
                if dens > best_key:
                    best_key, best = dens, c
                    best._tmp_val = val
                    best._tmp_tok = token

            if best is None:
                break
            if (self.budget_tokens is not None) and (spent_tokens + best._tmp_tok > self.budget_tokens):
                break

            best.value = max(0.0, min(1.0, float(best._tmp_val)))
            best.cost  = 1.0 - best.value
            selected.append(best)
            spent_tokens += int(best._tmp_tok)
            remain.remove(best)

        return selected
