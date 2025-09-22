class TmpGraph:
    def __init__(self):
        self.nodes = {}     # id -> node dict
        self.edges = []     # (u,v,type)
        self.used_cov = set()
        self.seen = set()
    def add_node(self, n:dict):
        self.nodes[n["id"]] = n
        self.used_cov |= set(n.get("cov", []))
        self.seen.add(n.get("text",""))
    def add_edge(self, u:str, v:str, etype:str):
        self.edges.append((u,v,etype))
    def snapshot(self)->dict:
        return {"n":len(self.nodes),"m":len(self.edges),"cov":len(self.used_cov)}
