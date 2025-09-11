import os
import sys

sys.path.append(r"F:\GSter")
from openai import OpenAI



def retrieve_astar_from_proto(texts, ts, query,
                              k=6, sim_th=0.30, alpha=0.15,
                              budget_nodes=3, tau=1.0):
    """texts: [str], ts: [int], query: str -> return path(list of idx)"""
    meta = [{"t": t} for t in ts]
    mg = MemoryGraph(texts, meta)
    mg.build_knn_edges(k=k, sim_th=sim_th)
    mg.build_temporal_edges(alpha=alpha)
    q_vec = mg.query_vector(query)
    path, _info = astar_search(mg, q_vec, budget_nodes=budget_nodes, tau=tau)
    return path


def retrieve_evidence_with_astar(query: str, memory_items):
    """
    memory_items: [{"text": "...", "t": 101}, ...]
    """
    texts = [m["text"] for m in memory_items]
    ts = [m.get("t", i) for i, m in enumerate(memory_items)]
    ids = retrieve_astar_from_proto(texts, ts, query,
                                    k=6, sim_th=0.30, budget_nodes=3, tau=1.0)
    evidence = [texts[i] for i in ids]
    return evidence, ids


def answer_with_evidence(query: str, evidence: list[str], model: str = "qwen-plus"):
    client = OpenAI(base_url=os.environ["OPENAI_API_BASE"],
                    api_key=os.environ["OPENAI_API_KEY"])
    prompt = "Use the ordered evidence to answer concisely.\n\n" + \
             "\n\n".join(f"[{i}] {e}" for i, e in enumerate(evidence))
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"Q: {query}\n\n{prompt}"}],
        max_tokens=256, temperature=0.2
    )
    return r.choices[0].message.content, getattr(r, "usage", {})
