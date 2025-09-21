import re, math
from typing import Dict, Any, Set
from .value_aware_astar import Candidate

def estimate_tokens(text:str)->int:
    return max(1, math.ceil(len(text)/4))  # 4字符≈1 token 的粗估

def extract_cov(text:str)->Set[str]:
    # 极简关键词抽取（可按需替换）
    toks = re.findall(r"[a-z]+(?:\s+\d+)?", text.lower())
    return set(toks)

def cdmem_to_candidate(obj:Dict[str,Any])->Candidate:
    """
    将 CDMem 的候选/记忆条目统一为 Candidate：
    支持字段：action / text / summary, dense_sim / bm25, id / uid / src_ids / cost_est
    """
    txt = obj.get("action") or obj.get("text") or obj.get("summary") or ""
    rel = obj.get("dense_sim", None)
    if rel is None: rel = obj.get("bm25", 0.0)
    cov = extract_cov(txt)
    cost = float(obj.get("cost_est") or estimate_tokens(txt))
    return Candidate(
        id=str(obj.get("id") or obj.get("uid") or hash(txt)),
        text=txt, rel=float(rel), cov=cov, cost_est=cost,
        meta={k:obj.get(k) for k in ("src_ids","bm25","dense_sim","type") if k in obj}
    )
