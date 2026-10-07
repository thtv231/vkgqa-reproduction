"""
Bước 2: Load MetaQA kb.txt → NetworkX DiGraph, lưu pickle để dùng lại.
Format mỗi dòng: subject|relation|object
"""

import pickle
from pathlib import Path

import networkx as nx

KB_PATH = Path(__file__).parent.parent / "MetaQA" / "kb" / "kb.txt"
GRAPH_PATH = Path(__file__).parent / "metaqa_kg.pkl"


def build_graph(kb_path: Path) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    with open(kb_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 3:
                continue
            subj, rel, obj = parts
            G.add_edge(subj.strip(), obj.strip(), relation=rel.strip())
    return G


def build_lookup(G: nx.MultiDiGraph) -> dict[str, str]:
    """lowercase → canonical entity name, để fuzzy match sau này."""
    return {n.lower(): n for n in G.nodes()}


def main():
    if not KB_PATH.exists():
        print(f"[ERROR] Không tìm thấy {KB_PATH}")
        print("Vui lòng download MetaQA kb.txt từ Google Drive:")
        print("https://drive.google.com/drive/folders/0B-36Uca2AvwhTWVFSUZqRXVtbUE")
        print(f"Và đặt tại: {KB_PATH}")
        return

    print(f"Loading {KB_PATH} ...")
    G = build_graph(KB_PATH)
    lookup = build_lookup(G)

    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")

    with open(GRAPH_PATH, "wb") as f:
        pickle.dump({"graph": G, "lookup": lookup}, f)

    print(f"Saved → {GRAPH_PATH}")

    # Relations
    rels = set(d["relation"] for _, _, d in G.edges(data=True))
    print(f"Relations ({len(rels)}): {sorted(rels)}")


if __name__ == "__main__":
    main()
