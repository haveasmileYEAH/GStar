# gstar/graphs/reuse_index.py
from collections import defaultdict
from typing import List, Dict, Any, Set, Iterable
from datetime import datetime, timedelta


class ReuseIndex:
    def __init__(self, nodes: List[dict], *, min_cov: int = 3, max_age_days: int = None):
        self.items = []
        now = datetime.utcnow()
        for n in nodes:
            cov = set(n.get("cov", []))
            txt = (n.get("text") or "").strip()
            ts = n.get("ts")
            if len(cov) < min_cov or not txt:
                continue
            if max_age_days is not None and ts:
                try:
                    dt = datetime.utcfromtimestamp(float(ts))
                    if (now - dt) > timedelta(days=max_age_days):
                        continue
                except Exception:
                    pass
            self.items.append({"text": txt, "cov": cov, "rel": float(n.get("rel", 0.0)), "ts": ts or 0})

    def find_superset_actions(self, q_cov: Set[str], admissible: Iterable[str]) -> List[dict]:
        """返回覆盖超集且在 admissible 集内的动作节点（按策略已过滤）"""
        adm = set(a.strip().lower() for a in admissible)
        hits = []
        for it in self.items:
            if it["text"].lower() in adm and q_cov.issubset(it["cov"]):
                hits.append(it)
        return hits

    @staticmethod
    def sort_hits(hits: List[dict], prefer: str = "recent") -> List[dict]:
        if prefer == "rel":
            return sorted(hits, key=lambda x: (-x["rel"], -float(x["ts"])))
        # default: recent first
        return sorted(hits, key=lambda x: -float(x["ts"]))
