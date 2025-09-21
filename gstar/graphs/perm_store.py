import os, json, time, uuid

class PermStore:
    def __init__(self, path="runs/perm_graph.jsonl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
    def upsert(self, nodes:list[dict]):
        with open(self.path, "a", encoding="utf-8") as f:
            for n in nodes:
                if "id" not in n: n["id"]=str(uuid.uuid4())
                n["ts"]=time.time()
                f.write(json.dumps(n, ensure_ascii=False)+"\n")
    def load_all(self):
        if not os.path.exists(self.path): return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(x) for x in f]
