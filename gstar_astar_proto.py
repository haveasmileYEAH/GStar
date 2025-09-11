import os, sys, json, math, heapq, argparse, textwrap, shutil
from typing import List, Dict, Any, Tuple
import numpy as np
from openai import OpenAI
from astar_adapter import retrieve_astar_from_proto
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def est_tokens(s: str) -> int:
    w = len(s.split())
    return max(1, int(round(w * 1.3)))


class MemoryGraph:
    def __init__(self, texts: List[str], meta: List[Dict[str, Any]]):
        self.texts = texts
        self.meta = meta
        self.N = len(texts)
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1)
        self.doc_mat = self.vectorizer.fit_transform(texts)  # 稀疏矩阵
        self.adj = [[] for _ in range(self.N)]  # 邻接表图结构

    def build_knn_edges(self, k: int = 8, sim_th: float = 0.20):
        """
        计算文档之间的余弦相似度矩阵，并使用knn计算 k 最相关。
        :param k: knn超参数
        :param sim_th: 相似度
        :return: NULL
        """
        sims = cosine_similarity(self.doc_mat)
        kk = max(1, min(k, self.N - 1))
        for i in range(self.N):
            idx = np.argpartition(-sims[i], kk)[:kk + 1]
            for j in idx:
                if j == i:
                    continue
                if sims[i, j] >= sim_th:
                    self.adj[i].append((j, 1.0 - float(sims[i, j])))

    def build_temporal_edges(self, alpha: float = 0.15):
        """
        按时间排序，将相邻节点联系起来，构建记忆图
        :param alpha: 超参数，
        :return:
        """
        order = sorted(range(self.N), key=lambda x: self.meta[x].get("t", x))
        for a, b in zip(order, order[1:]):
            self.adj[a].append((b, alpha))

    def query_vector(self, q: str):
        return self.vectorizer.transform([q])

    def sims_to_query(self, q_vec):
        sims = cosine_similarity(self.doc_mat, q_vec).reshape(-1)
        return sims

    # 求查询需求和每一个节点之间的余弦相似度

def astar_search(graph: MemoryGraph, q_vec, budget_nodes: int = 6, tau: float = 1.6,
                 rep_lambda: float = 0.15) -> Tuple[List[int], Dict[str, Any]]:
    sims = graph.sims_to_query(q_vec)
    start = int(np.argmax(sims))

    # 启发函数，1 - 相似度，相似度越接近，结果越小
    def h(node_idx: int) -> float:
        return max(0.0, 1.0 - float(sims[node_idx]))

    used_idxs = set()
    start_cost = est_tokens(graph.texts[start])
    start_sim = float(sims[start])

    # 实际代价： 边代价，重复惩罚，token成本
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
            rep_pen = rep_lambda if v in used else 0.0
            add_tokens = est_tokens(graph.texts[v])
            g2 = g + edge_cost + rep_pen + 0.01 * add_tokens
            if v not in visited_best_g or g2 < visited_best_g[v] - 1e-9:
                visited_best_g[v] = g2
                used2 = set(used)
                if v not in used2:
                    used2.add(v)
                sum_sims2 = sum_sims + float(sims[v]) if v not in path else sum_sims
                total_tokens2 = total_tokens + add_tokens if v not in path else total_tokens
                f2 = g2 + h(v)
                heapq.heappush(openpq, (f2, g2, v, path + [v], used2, sum_sims2, total_tokens2))
    return best_path, best_info


def summarize_path(texts: List[str], path: List[int], query: str, model: str = "qwen-plus") -> str:
    base = os.environ.get("OPENAI_API_BASE")
    key = os.environ.get("OPENAI_API_KEY")
    if not base or not key:
        return "(LLM OFF) Set OPENAI_API_BASE & OPENAI_API_KEY to enable Qwen Intl summarization."
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
            model=model,
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
    ap.add_argument("--model", type=str, default="qwen-plus", help="LLM model name if calling OpenAI-compatible API")
    ap.add_argument("--llm", action="store_true", help="call LLM to summarize/answer")
    args = ap.parse_args()
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
            texts.append(s)
            meta.append({"t": t})
    mg = MemoryGraph(texts, meta)
    mg.build_knn_edges(k=args.k, sim_th=args.sim_th)
    mg.build_temporal_edges(alpha=args.alpha)
    q_vec = mg.query_vector(args.query)
    path, info = astar_search(mg, q_vec, budget_nodes=args.budget_nodes, tau=args.tau)
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


if __name__ == "__main__":
    main()
