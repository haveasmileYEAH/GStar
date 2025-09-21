#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GStar A* Retrieval (GoT-style edge building + Value-aware edges)
"""

import os, json, heapq, argparse
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from got_ops import TFIDFGoTPolicy
import json, time, re
from typing import Tuple


def est_tokens(s: str) -> int:
    w = len(s.split())
    return max(1, int(round(w * 1.3)))


class MemoryGraph:
    def __init__(self, texts: List[str], meta: List[Dict[str, Any]]):
        self.texts = texts
        self.meta = meta
        self.N = len(texts)
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1)
        self.doc_mat = self.vectorizer.fit_transform(texts)
        self.adj = [[] for _ in range(self.N)]
        self.dd_sims = None  # 文档↔文档相似度矩阵（供 A* 冗余惩罚）

    def build_temporal_edges(self, alpha: float = 0.15):
        order = sorted(range(self.N), key=lambda x: self.meta[x].get("t", x))
        for a, b in zip(order, order[1:]):
            self.adj[a].append((b, alpha))

    def query_vector(self, q: str):
        return self.vectorizer.transform([q])

    def sims_to_query(self, q_vec):
        return cosine_similarity(self.doc_mat, q_vec).reshape(-1)

    def build_edges_with_policy(self, policy) -> None:
        self.adj = [[] for _ in range(self.N)]
        for i in range(self.N):
            cands = list(policy.generate(i))
            policy.score(i, cands)
            chosen = policy.select(i, cands)
            for c in chosen:
                self.adj[i].append((c.j, c.cost))
        # 把 dd 传出来，供 A* 的冗余惩罚使用
        if hasattr(policy, "dd"):
            self.dd_sims = policy.dd


def astar_search(graph: MemoryGraph, q_vec,
                 budget_nodes: int = 6, tau: float = 1.6,
                 rep_lambda: float = 0.15,
                 gamma_q: float = 0.35,  # 查询不匹配惩罚
                 rho_path: float = 0.25  # 路径内冗余惩罚
                 ) -> Tuple[List[int], Dict[str, Any]]:
    sims = graph.sims_to_query(q_vec)  # 文档↔查询
    start = int(np.argmax(sims))

    dd = graph.dd_sims
    if dd is None:
        dd = cosine_similarity(graph.doc_mat)

    def h(node_idx: int) -> float:
        return max(0.0, 1.0 - float(sims[node_idx]))

    start_cost = est_tokens(graph.texts[start])
    start_sim = float(sims[start])

    openpq = [(start_cost + h(start), start_cost, start, [start], {start}, start_sim, start_cost)]
    visited_best_g = {start: start_cost}

    best_path = [start]
    best_info = {"sum_sims": start_sim, "total_tokens": start_cost}

    while openpq:
        f, g, u, path, used, sum_sims, total_tokens = heapq.heappop(openpq)
        if sum_sims >= tau or len(path) >= budget_nodes:
            best_path, best_info = path, {"sum_sims": sum_sims, "total_tokens": total_tokens}
            break

        for v, edge_cost in graph.adj[u]:
            if v in path:
                continue
            # 查询不匹配惩罚（越不贴，越罚）
            q_pen = gamma_q * (1.0 - float(sims[v]))
            # 路径冗余惩罚（越像已选，越罚）
            if path:
                max_sim = float(np.max(dd[v, np.array(path, dtype=int)]))
            else:
                max_sim = 0.0
            mmr_pen = rho_path * max_sim

            rep_pen = rep_lambda if v in used else 0.0
            add_tokens = est_tokens(graph.texts[v])
            g2 = g + edge_cost + rep_pen + 0.01 * add_tokens + q_pen + mmr_pen

            if v not in visited_best_g or g2 < visited_best_g[v] - 1e-9:
                visited_best_g[v] = g2
                used2 = set(used);
                used2.add(v)
                sum_sims2 = sum_sims + float(sims[v])
                total_tokens2 = total_tokens + add_tokens
                f2 = g2 + h(v)
                if sum_sims2 > best_info["sum_sims"] + 1e-9:
                    best_path = path + [v]
                    best_info = {"sum_sims": sum_sims2, "total_tokens": total_tokens2}
                heapq.heappush(openpq, (f2, g2, v, path + [v], used2, sum_sims2, total_tokens2))

    return best_path, best_info


def summarize_path(texts: List[str], path: List[int], query: str, model: str = "qwen-plus") -> str:
    # === updated: accept OPENAI_*, QWEN_*, TOGETHER_* ===
    base_env, key_env, model_env = _get_openai_compat_from_env()
    base = base_env or os.environ.get("OPENAI_API_BASE")
    key = key_env or os.environ.get("OPENAI_API_KEY")
    mdl = model_env or model
    if not base or not key:
        return "(LLM OFF) Set OPENAI_API_BASE/KEY or QWEN_BASE_URL/DASHSCOPE_API_KEY (or TOGETHER_*) to enable summarization."
    try:
        from openai import OpenAI
    except Exception:
        return "(LLM OFF) openai package not installed. pip install openai"
    client = OpenAI(base_url=base, api_key=key)
    evidence = "\n\n".join([f"[{i}] {texts[i][:300]}" for i in path])
    prompt = f"""You are given a user query and a chain of evidence (ordered).
Answer the query concisely, citing the most relevant evidence indices in [] if helpful.
Query: {query}

Evidence:
{evidence}
"""
    try:
        r = client.chat.completions.create(
            model=mdl,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.2,
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"(LLM OFF) Error calling LLM: {e}"



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="", help="path to jsonl with fields: {text, t}")
    ap.add_argument("--query", type=str, default="Put the apple into the fridge, what should I do?")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--sim_th", type=float, default=0.20)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--budget_nodes", type=int, default=6)
    ap.add_argument("--tau", type=float, default=1.6)
    # GoT/价值剪枝参数
    ap.add_argument("--edge_select", type=str, default="value", choices=["value", "topk"])
    ap.add_argument("--rho", type=float, default=0.35, help="Base redundancy weight (used if auto_rho=False)")
    ap.add_argument("--lam_cost", type=float, default=0.10, help="Token cost penalty in value pruning")
    ap.add_argument("--edge_budget_tokens", type=int, default=0, help="Per-node edge token budget; 0=unlimited")
    ap.add_argument("--auto_rho", type=int, default=1, help="1: enable per-node adaptive rho; 0: fixed rho")
    ap.add_argument("--rho_min", type=float, default=0.30)
    ap.add_argument("--rho_max", type=float, default=0.70)
    # A* 正则
    ap.add_argument("--gamma_q", type=float, default=0.35, help="Query mismatch penalty in A*")
    ap.add_argument("--rho_path", type=float, default=0.25, help="Path redundancy penalty in A*")
    # LLM
    ap.add_argument("--model", type=str, default="qwen-plus")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--edge_gamma_q", type=float, default=0.05, help="Edge-level query mismatch penalty (gentle)")
    ap.add_argument("--edge_q_floor", type=float, default=0.10, help="Floor for sim_q in edge-level penalty")
    args = ap.parse_args()

    # 1) 读数据
    texts, meta = [], []
    if args.data and os.path.exists(args.data):
        with open(args.data, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                texts.append(obj["text"])
                meta.append({"t": obj.get("t", len(meta))})
    else:
        toy = [
            ("You are in the kitchen. The fridge is closed. There is an apple on the counter next to a knife.", 1),
            ("Open the fridge first before placing any item inside. The fridge has a shelf for produce.", 2),
            ("The living room contains a sofa and a TV. A remote lies on the table.", 3),
            (
            "To put the apple into the fridge: pick up the apple, open the fridge door, place the apple on the top shelf.",
            4),
            ("Microwave usage: never put metallic objects inside. Use 1 minute to reheat leftovers.", 5),
            ("After placing items in the fridge, close the door to keep temperature stable.", 6),
            ("Bedroom notes: closet is on the left, window on the right.", 7),
            ("If the fridge door is stuck, pull the handle firmly. Do not force beyond normal resistance.", 8),
        ]
        for s, t in toy:
            texts.append(s);
            meta.append({"t": t})

    mg = MemoryGraph(texts, meta)

    # 2) 计算与查询的相似度（TF-IDF），并归一化到 [0,1]
    q_vec = mg.query_vector(args.query)
    sim_q = mg.sims_to_query(q_vec)
    sim_q = (sim_q - sim_q.min()) / (sim_q.max() - sim_q.min() + 1e-9)

    # 3) GoT 策略建图（价值剪枝 + 自适应 ρ）
    token_costs = np.array([est_tokens(t) for t in texts], dtype=np.float32)
    policy = TFIDFGoTPolicy(
        doc_mat=mg.doc_mat, sim_q=sim_q,
        k_doc=args.k, kq=max(1, args.k // 2),
        q_lambda=0.55, sim_mode="pct", sim_th=args.sim_th, sim_pct=0.80,
        keep_n=max(args.k, 6), ensure_deg=2,
        select_mode=args.edge_select,
        token_costs=token_costs,
        rho=args.rho, lam_cost=args.lam_cost,
        budget_tokens=(args.edge_budget_tokens if args.edge_budget_tokens > 0 else None),
        auto_rho=bool(args.auto_rho), rho_min=args.rho_min, rho_max=args.rho_max,
        edge_gamma_q=args.edge_gamma_q, edge_q_floor=args.edge_q_floor,
    )

    mg.build_edges_with_policy(policy)

    # 4) 时间边：仅当 alpha>0 时添加
    if args.alpha > 0:
        mg.build_temporal_edges(alpha=args.alpha)

    # 5) 调试：查看节点 3 的邻居
    i = 3
    sims_doc = cosine_similarity(mg.doc_mat)
    nbrs = sorted(mg.adj[i], key=lambda x: x[1])[:10]
    print(f"\n[DEBUG] Neighbors of node {i}:")
    for j, cost in nbrs:
        edge_type = "temporal" if args.alpha > 0 and abs(cost - args.alpha) < 1e-9 else "knn"
        sim_ij = float(sims_doc[i, j])
        print(f"  {i} -> {j}  edge={edge_type:8s}  edge_cost={cost:.3f}  sim(doc{i},doc{j})={sim_ij:.3f}")

    # 6) A* 搜索（含查询/冗余软约束）
    path, info = astar_search(
        mg, q_vec,
        budget_nodes=args.budget_nodes, tau=args.tau,
        gamma_q=args.gamma_q, rho_path=args.rho_path
    )

    print("=== GStar A* Retrieval Prototype ===")
    print(f"Query: {args.query}")
    print(f"Path (node indices): {path}")
    print(f"Sum of similarities (evidence): {info['sum_sims']:.3f}")
    print(f"Estimated total tokens: {info['total_tokens']}")

    print("\n--- Evidence chain ---")
    for idx in path:
        t = texts[idx].replace("\n", " ")
        print(f"[{idx}] {t[:120]}{'...' if len(t) > 120 else ''}")

    if args.llm:
        print("\n--- LLM Answer (Qwen Intl via OpenAI-compatible API) ---")
        ans = summarize_path(texts, path, args.query, model=args.model)
        print(ans)

# === NEW: choose one admissible action using your GStar (GoT+A*) ===
def choose_action_with_gstar(obs: str,
                             admissible: List[str],
                             goal: str = "",
                             k: int = 8,
                             tau: float = 1.05,
                             budget_nodes: int = 3,
                             edge_alpha: float = 0.0) -> str:
    """
    Use your MemoryGraph + TFIDFGoTPolicy + A* to pick EXACTLY ONE action from `admissible`.
    ALFWorld is English-only; `obs` / `admissible` / `goal` should be English strings.
    """
    # 0) guard
    if not admissible:
        return "look"

    # 1) Build a tiny graph on candidate actions
    texts = list(dict.fromkeys(admissible))  # de-dup, keep order
    meta = [{"t": i} for i in range(len(texts))]

    mg = MemoryGraph(texts, meta)

    # 2) Query = goal + observation
    query = (goal.strip() + "\n" + obs.strip()).strip()
    q_vec = mg.query_vector(query)
    sim_q = mg.sims_to_query(q_vec)
    sim_q = (sim_q - sim_q.min()) / (sim_q.max() - sim_q.min() + 1e-9)

    # 3) Build edges with your GoT policy (value-aware)
    token_costs = np.array([est_tokens(t) for t in texts], dtype=np.float32)
    k_doc = min(max(3, k), len(texts))
    policy = TFIDFGoTPolicy(
        doc_mat=mg.doc_mat, sim_q=sim_q,
        k_doc=k_doc, kq=max(1, min(3, k_doc)),
        q_lambda=0.55, sim_mode="pct", sim_th=0.0, sim_pct=0.80,
        keep_n=len(texts), ensure_deg=1,
        select_mode="value",
        token_costs=token_costs,
        rho=0.35, lam_cost=0.05,
        budget_tokens=None,
        auto_rho=True, rho_min=0.30, rho_max=0.70,
        edge_gamma_q=0.05, edge_q_floor=0.10,
    )
    mg.build_edges_with_policy(policy)
    if edge_alpha > 0:
        mg.build_temporal_edges(alpha=edge_alpha)

    # 4) A* over the action graph; we only need the best-first node as the action
    path, info = astar_search(
        mg, mg.query_vector(query),
        budget_nodes=max(1, budget_nodes), tau=tau,
        gamma_q=0.35, rho_path=0.25
    )
    best_idx = path[0] if path else int(np.argmax(sim_q))
    return texts[best_idx]


# === PATCH: make summarize_path read your current env vars too (backward-compatible) ===
def _get_openai_compat_from_env():
    """
    Return (base_url, api_key, model) with compatibility for OPENAI_*, QWEN_*, TOGETHER_*.
    """
    base = os.environ.get("OPENAI_API_BASE") \
        or os.environ.get("QWEN_BASE_URL") \
        or os.environ.get("TOGETHER_BASE_URL")
    key = os.environ.get("OPENAI_API_KEY") \
        or os.environ.get("DASHSCOPE_API_KEY") \
        or os.environ.get("TOGETHER_API_KEY")
    model = os.environ.get("OPENAI_MODEL") \
        or os.environ.get("QWEN_MODEL") \
        or os.environ.get("TOGETHER_MODEL") \
        or "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    return base, key, model

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _sent_split(text: str) -> list[str]:
    # 简单句子切分：句号/问号/换行；可按需替换成更强的分句器
    text = (text or "").strip()
    parts = re.split(r"(?<=[\.\?\!])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]

def _common_prefix_len(a_tokens: list[str], b_tokens: list[str]) -> int:
    n = min(len(a_tokens), len(b_tokens))
    i = 0
    while i < n and a_tokens[i] == b_tokens[i]:
        i += 1
    return i

class RAGMemory:
    """
    以 JSONL 方式存储问答记忆（追加写入，按 qid 去重折叠）。
    每条item结构：
      {"qid": "...", "q": "...", "a": "...", "segments": [{"text": "...","t": 1234}, ...], "t": 1234}
    """
    def __init__(self, path: str = "data/memory.jsonl"):
        self.path = path
        self.items = []          # 折叠后的最新视图
        self._by_qid = {}        # qid -> item
        self._load()

    def _load(self):
        self.items, self._by_qid = [], {}
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        it = json.loads(line)
                    except Exception:
                        continue
                    qid = it.get("qid") or _norm(it.get("q",""))
                    self._by_qid[qid] = it  # 以最后一次为准（折叠）
        self.items = list(self._by_qid.values())

    def _save_replace(self, item: dict):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # 以“更新”事件追加，读取时按 qid 折叠到最新
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        self._by_qid[item["qid"]] = item
        self.items = list(self._by_qid.values())

    def exact_match(self, q: str):
        nq = _norm(q)
        for it in self.items:
            if _norm(it.get("q","")) == nq:
                return it
        return None

    def partial_match(self, q: str, threshold: float = 0.88) -> Tuple[dict, float]:
        # 用 TF-IDF 做简易相似度检索
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        qs = [_norm(it.get("q","")) for it in self.items]
        if not qs:
            return None, 0.0
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
        X = vec.fit_transform(qs)
        v = vec.transform([_norm(q)])
        sims = cosine_similarity(X, v).reshape(-1)
        i = int(sims.argmax())
        sc = float(sims[i])
        return (self.items[i], sc) if sc >= threshold else (None, sc)

    def upsert_with_breakpoint(self, q: str, new_full_answer: str, base_qid: str | None):
        """
        把 new_full_answer 合并入 base_qid（若存在），只在“断点”之后追加 segments；
        若找不到 base_qid，则创建新item。
        """
        now = int(time.time())
        if base_qid and base_qid in self._by_qid:
            item = self._by_qid[base_qid]
        else:
            # 新建
            qid = _norm(q)
            segs = [{"text": s, "t": now} for s in _sent_split(new_full_answer)]
            item = {"qid": qid, "q": q, "a": new_full_answer, "segments": segs, "t": now}
            self._save_replace(item)
            return item

        old_ans = item.get("a","")
        old_tok = old_ans.split()
        new_tok = (new_full_answer or "").split()
        bp = _common_prefix_len(old_tok, new_tok)  # 断点位置（按词）

        # 仅追加“断点之后”的内容
        new_tail = " ".join(new_tok[bp:]).strip()
        if not new_tail:
            return item  # 无新增

        # 更新全文
        merged = (old_ans + (" " if old_ans and new_tail else "") + new_tail).strip()
        # 生成新的段落/句子分段（只对 tail 切分）
        new_segments = [{"text": s, "t": now} for s in _sent_split(new_tail)]
        segs = (item.get("segments") or []) + new_segments

        new_item = {"qid": item["qid"], "q": item.get("q", q), "a": merged, "segments": segs, "t": now}
        self._save_replace(new_item)
        return new_item

def _answer_via_gstar_llm(query: str, memory_items: list[dict]) -> str:
    """
    用你现有的 GStar 检索 + summarize_path 生成完整新答案。
    把记忆的 Q/A 作为“证据文本”构图检索，然后走 summarize_path。
    """
    texts, meta = [], []
    for i, it in enumerate(memory_items):
        txt = (f"Q: {it.get('q','')}\nA: {it.get('a','')}").strip()
        texts.append(txt); meta.append({"t": it.get("t", i)})

    mg = MemoryGraph(texts if texts else ["(empty)"], meta if meta else [{"t": 0}])  # <-- 你现有的类
    q_vec = mg.query_vector(query)
    sim_q = mg.sims_to_query(q_vec)
    sim_q = (sim_q - sim_q.min()) / (sim_q.max() - sim_q.min() + 1e-9)

    token_costs = np.array([est_tokens(t) for t in texts or ["(empty)"]], dtype=np.float32)
    policy = TFIDFGoTPolicy(
        doc_mat=mg.doc_mat, sim_q=sim_q,
        k_doc=min(8, len(texts) or 1), kq=3,
        q_lambda=0.55, sim_mode="pct", sim_th=0.20, sim_pct=0.80,
        keep_n=max(6, min(8, len(texts) or 1)), ensure_deg=2,
        select_mode="value", token_costs=token_costs,
        rho=0.35, lam_cost=0.05, budget_tokens=None,
        auto_rho=True, rho_min=0.30, rho_max=0.70,
        edge_gamma_q=0.05, edge_q_floor=0.10,
    )
    mg.build_edges_with_policy(policy)
    path, _info = astar_search(mg, q_vec, budget_nodes=6, tau=1.6, gamma_q=0.35, rho_path=0.25)
    return summarize_path(texts or ["(empty)"], path, query)

def answer_with_memory_router_bp(query: str, memory_path: str = "data/memory.jsonl",
                                 partial_th: float = 0.88) -> Tuple[str, dict]:
    """
    三段式 + 断点合并：
      1) 完全相同：直接返回旧答（零LLM）
      2) 部分相同/不相同：先用旧答作部分记忆，再生成完整新答；只把“断点后的尾部”并入同一条记忆
    返回：(answer, meta)
    """
    mem = RAGMemory(memory_path)

    # 1) exact
    hit = mem.exact_match(query)
    if hit:
        return hit.get("a",""), {"route": "exact", "qid": hit.get("qid")}

    # 2) partial（>=阈值） or novel（<阈值）
    near, score = mem.partial_match(query, threshold=partial_th)
    base_qid = near.get("qid") if near else None

    # 用“部分记忆/全部记忆”作为证据构图，让 LLM 生成完整新答案
    # （你也可以换成只生成补全尾部的提示，这里保持稳妥：先拿全新答案，再做断点合并）
    ans_full = _answer_via_gstar_llm(query, mem.items if mem.items else [])

    # 断点合并到“同一条记忆”（不是新建）
    item = mem.upsert_with_breakpoint(q=query, new_full_answer=ans_full, base_qid=base_qid)
    return item.get("a",""), {"route": ("partial" if near else "novel"), "near_score": score, "qid": item.get("qid")}

# === NEW: choose one admissible action using your GStar (GoT+A*) ===
def choose_action_with_gstar(obs: str,
                             admissible: List[str],
                             goal: str = "",
                             k: int = 8,
                             tau: float = 1.05,
                             budget_nodes: int = 3,
                             edge_alpha: float = 0.0) -> str:
    """
    Use your MemoryGraph + TFIDFGoTPolicy + A* to pick EXACTLY ONE action from `admissible`.
    ALFWorld is English-only; `obs` / `admissible` / `goal` should be English strings.
    """
    # 0) guard
    if not admissible:
        return "look"

    # 1) Build a tiny graph on candidate actions
    texts = list(dict.fromkeys(admissible))  # de-dup, keep order
    meta = [{"t": i} for i in range(len(texts))]

    mg = MemoryGraph(texts, meta)

    # 2) Query = goal + observation
    query = (goal.strip() + "\n" + obs.strip()).strip()
    q_vec = mg.query_vector(query)
    sim_q = mg.sims_to_query(q_vec)
    sim_q = (sim_q - sim_q.min()) / (sim_q.max() - sim_q.min() + 1e-9)

    # 3) Build edges with your GoT policy (value-aware)
    token_costs = np.array([est_tokens(t) for t in texts], dtype=np.float32)
    k_doc = min(max(3, k), len(texts))
    policy = TFIDFGoTPolicy(
        doc_mat=mg.doc_mat, sim_q=sim_q,
        k_doc=k_doc, kq=max(1, min(3, k_doc)),
        q_lambda=0.55, sim_mode="pct", sim_th=0.0, sim_pct=0.80,
        keep_n=len(texts), ensure_deg=1,
        select_mode="value",
        token_costs=token_costs,
        rho=0.35, lam_cost=0.05,
        budget_tokens=None,
        auto_rho=True, rho_min=0.30, rho_max=0.70,
        edge_gamma_q=0.05, edge_q_floor=0.10,
    )
    mg.build_edges_with_policy(policy)
    if edge_alpha > 0:
        mg.build_temporal_edges(alpha=edge_alpha)

    # 4) A* over the action graph; we only need the best-first node as the action
    path, info = astar_search(
        mg, mg.query_vector(query),
        budget_nodes=max(1, budget_nodes), tau=tau,
        gamma_q=0.35, rho_path=0.25
    )
    best_idx = path[0] if path else int(np.argmax(sim_q))
    return texts[best_idx]


# === PATCH: make summarize_path read your current env vars too (backward-compatible) ===
def _get_openai_compat_from_env():
    """
    Return (base_url, api_key, model) with compatibility for OPENAI_*, QWEN_*, TOGETHER_*.
    """
    base = os.environ.get("OPENAI_API_BASE") \
           or os.environ.get("QWEN_BASE_URL") \
           or os.environ.get("TOGETHER_BASE_URL")
    key = os.environ.get("OPENAI_API_KEY") \
          or os.environ.get("DASHSCOPE_API_KEY") \
          or os.environ.get("TOGETHER_API_KEY")
    model = os.environ.get("OPENAI_MODEL") \
            or os.environ.get("QWEN_MODEL") \
            or os.environ.get("TOGETHER_MODEL") \
            or "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    return base, key, model


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _sent_split(text: str) -> list[str]:
    # 简单句子切分：句号/问号/换行；可按需替换成更强的分句器
    text = (text or "").strip()
    parts = re.split(r"(?<=[\.\?\!])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _common_prefix_len(a_tokens: list[str], b_tokens: list[str]) -> int:
    n = min(len(a_tokens), len(b_tokens))
    i = 0
    while i < n and a_tokens[i] == b_tokens[i]:
        i += 1
    return i


class RAGMemory:
    """
    以 JSONL 方式存储问答记忆（追加写入，按 qid 去重折叠）。
    每条item结构：
      {"qid": "...", "q": "...", "a": "...", "segments": [{"text": "...","t": 1234}, ...], "t": 1234}
    """

    def __init__(self, path: str = "data/memory.jsonl"):
        self.path = path
        self.items = []  # 折叠后的最新视图
        self._by_qid = {}  # qid -> item
        self._load()

    def _load(self):
        self.items, self._by_qid = [], {}
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        it = json.loads(line)
                    except Exception:
                        continue
                    qid = it.get("qid") or _norm(it.get("q", ""))
                    self._by_qid[qid] = it  # 以最后一次为准（折叠）
        self.items = list(self._by_qid.values())

    def _save_replace(self, item: dict):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # 以“更新”事件追加，读取时按 qid 折叠到最新
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        self._by_qid[item["qid"]] = item
        self.items = list(self._by_qid.values())

    def exact_match(self, q: str):
        nq = _norm(q)
        for it in self.items:
            if _norm(it.get("q", "")) == nq:
                return it
        return None

    def partial_match(self, q: str, threshold: float = 0.88) -> Tuple[dict, float]:
        # 用 TF-IDF 做简易相似度检索
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        qs = [_norm(it.get("q", "")) for it in self.items]
        if not qs:
            return None, 0.0
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        X = vec.fit_transform(qs)
        v = vec.transform([_norm(q)])
        sims = cosine_similarity(X, v).reshape(-1)
        i = int(sims.argmax())
        sc = float(sims[i])
        return (self.items[i], sc) if sc >= threshold else (None, sc)

    def upsert_with_breakpoint(self, q: str, new_full_answer: str, base_qid: str | None):
        """
        把 new_full_answer 合并入 base_qid（若存在），只在“断点”之后追加 segments；
        若找不到 base_qid，则创建新item。
        """
        now = int(time.time())
        if base_qid and base_qid in self._by_qid:
            item = self._by_qid[base_qid]
        else:
            # 新建
            qid = _norm(q)
            segs = [{"text": s, "t": now} for s in _sent_split(new_full_answer)]
            item = {"qid": qid, "q": q, "a": new_full_answer, "segments": segs, "t": now}
            self._save_replace(item)
            return item

        old_ans = item.get("a", "")
        old_tok = old_ans.split()
        new_tok = (new_full_answer or "").split()
        bp = _common_prefix_len(old_tok, new_tok)  # 断点位置（按词）

        # 仅追加“断点之后”的内容
        new_tail = " ".join(new_tok[bp:]).strip()
        if not new_tail:
            return item  # 无新增

        # 更新全文
        merged = (old_ans + (" " if old_ans and new_tail else "") + new_tail).strip()
        # 生成新的段落/句子分段（只对 tail 切分）
        new_segments = [{"text": s, "t": now} for s in _sent_split(new_tail)]
        segs = (item.get("segments") or []) + new_segments

        new_item = {"qid": item["qid"], "q": item.get("q", q), "a": merged, "segments": segs, "t": now}
        self._save_replace(new_item)
        return new_item


def _answer_via_gstar_llm(query: str, memory_items: list[dict]) -> str:
    """
    用你现有的 GStar 检索 + summarize_path 生成完整新答案。
    把记忆的 Q/A 作为“证据文本”构图检索，然后走 summarize_path。
    """
    texts, meta = [], []
    for i, it in enumerate(memory_items):
        txt = (f"Q: {it.get('q', '')}\nA: {it.get('a', '')}").strip()
        texts.append(txt);
        meta.append({"t": it.get("t", i)})

    mg = MemoryGraph(texts if texts else ["(empty)"], meta if meta else [{"t": 0}])  # <-- 你现有的类
    q_vec = mg.query_vector(query)
    sim_q = mg.sims_to_query(q_vec)
    sim_q = (sim_q - sim_q.min()) / (sim_q.max() - sim_q.min() + 1e-9)

    token_costs = np.array([est_tokens(t) for t in texts or ["(empty)"]], dtype=np.float32)
    policy = TFIDFGoTPolicy(
        doc_mat=mg.doc_mat, sim_q=sim_q,
        k_doc=min(8, len(texts) or 1), kq=3,
        q_lambda=0.55, sim_mode="pct", sim_th=0.20, sim_pct=0.80,
        keep_n=max(6, min(8, len(texts) or 1)), ensure_deg=2,
        select_mode="value", token_costs=token_costs,
        rho=0.35, lam_cost=0.05, budget_tokens=None,
        auto_rho=True, rho_min=0.30, rho_max=0.70,
        edge_gamma_q=0.05, edge_q_floor=0.10,
    )
    mg.build_edges_with_policy(policy)
    path, _info = astar_search(mg, q_vec, budget_nodes=6, tau=1.6, gamma_q=0.35, rho_path=0.25)
    return summarize_path(texts or ["(empty)"], path, query)


def answer_with_memory_router_bp(query: str, memory_path: str = "data/memory.jsonl",
                                 partial_th: float = 0.88) -> Tuple[str, dict]:
    """
    三段式 + 断点合并：
      1) 完全相同：直接返回旧答（零LLM）
      2) 部分相同/不相同：先用旧答作部分记忆，再生成完整新答；只把“断点后的尾部”并入同一条记忆
    返回：(answer, meta)
    """
    mem = RAGMemory(memory_path)

    # 1) exact
    hit = mem.exact_match(query)
    if hit:
        return hit.get("a", ""), {"route": "exact", "qid": hit.get("qid")}

    # 2) partial（>=阈值） or novel（<阈值）
    near, score = mem.partial_match(query, threshold=partial_th)
    base_qid = near.get("qid") if near else None

    # 用“部分记忆/全部记忆”作为证据构图，让 LLM 生成完整新答案
    # （你也可以换成只生成补全尾部的提示，这里保持稳妥：先拿全新答案，再做断点合并）
    ans_full = _answer_via_gstar_llm(query, mem.items if mem.items else [])

    # 断点合并到“同一条记忆”（不是新建）
    item = mem.upsert_with_breakpoint(q=query, new_full_answer=ans_full, base_qid=base_qid)
    return item.get("a", ""), {"route": ("partial" if near else "novel"), "near_score": score, "qid": item.get("qid")}


if __name__ == "__main__":
    main()
