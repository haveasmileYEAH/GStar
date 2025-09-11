# gstar_retriever.py —— 极简A*检索（英文TF-IDF）
from typing import List
import numpy as np, heapq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_astar(texts: List[str], ts: List[int], query: str,
                   k:int=8, sim_th:float=0.25, alpha:float=0.15,
                   budget_nodes:int=6, tau:float=1.6) -> list[int]:
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    X = vec.fit_transform(texts)
    sims = cosine_similarity(X, vec.transform([query])).reshape(-1)
    N = len(texts)

    # KNN语义边
    A = [[] for _ in range(N)]
    full = cosine_similarity(X)
    kk = max(1, min(k, N-1))
    for i in range(N):
        idx = np.argpartition(-full[i], kk)[:kk+1]
        for j in idx:
            if j == i: continue
            if full[i, j] >= sim_th:
                A[i].append((j, 1.0 - float(full[i, j])))

    # 时间边
    order = sorted(range(N), key=lambda i: ts[i])
    for a, b in zip(order, order[1:]):
        A[a].append((b, alpha))

    # A*
    start = int(np.argmax(sims))
    def h(u): return max(0.0, 1.0 - float(sims[u]))
    openpq = [(h(start), 0.0, start, [start], float(sims[start]))]  # (f,g,u,path,sum_sims)
    best_path = [start]

    while openpq:
        f, g, u, path, sum_sims = heapq.heappop(openpq)
        if sum_sims >= tau or len(path) >= budget_nodes:
            best_path = path; break
        for v, w in A[u]:
            if v in path: continue
            g2 = g + w
            heapq.heappush(openpq, (g2 + h(v), g2, v, path + [v], sum_sims + float(sims[v])))
    return best_path
