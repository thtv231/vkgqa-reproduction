"""
Bước 7: Phát lại bước trích subgraph (không render) để đo thống kê của từng chiến lược.

Với cùng --seed và --hop-filter như lúc chạy 03_generate_images.py, thứ tự gọi random
giống hệt nên tập node/edge thu được trùng với ảnh đã sinh.

Điều kiện *-legacy dùng cách tra cứu entity cũ (trước khi sửa lỗi trùng tên hoa/thường).

Đo cho mỗi câu hỏi: số node, số edge, đáp án có nằm trong subgraph không,
và có đường đi center→answer (≤ k bước, vô hướng) bên trong subgraph không.

Cách dùng:
  python 07_analyze_subgraphs.py --hop-filter 2 --out results/subgraph_stats_hop2.json
"""

import argparse
import importlib.util
import json
import random
from pathlib import Path

import networkx as nx

SCRIPTS_DIR = Path(__file__).parent
spec = importlib.util.spec_from_file_location("gen", SCRIPTS_DIR / "03_generate_images.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


def load_records(hop_filter: int) -> list[dict]:
    seen = {}
    with open(gen.SUBSET_JSONL, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            seen.setdefault(rec["image"], rec)
    records = list(seen.values())
    if hop_filter:
        records = [r for r in records if r["hop_k"] == hop_filter]
    return records


def replay(G, lookup, records, strategy: int, directed: bool, seed: int, legacy: bool) -> list[dict]:
    random.seed(seed)
    gen.DIRECTED_PATH = directed
    gen.LEGACY_RESOLVE = legacy
    out = []
    for rec in records:
        center = gen.resolve_entity(rec["entity_name"], lookup)
        if center is None:
            continue
        answer = gen.resolve_entity(str(rec.get("answer", "")).split("|")[0].strip(), lookup)
        path_found = None
        if strategy == 2 and answer and rec.get("category", "") == "hop reasoning":
            path_found = gen.find_reasoning_path(G, center, answer, max_k=rec["hop_k"]) is not None
            sub, _ = gen.extract_subgraph_strategy2(G, center, answer, rec["hop_k"])
        else:
            sub, _ = gen.extract_subgraph(G, center, rec["hop_k"])
        has_answer = answer is not None and answer in sub
        reachable = False
        if has_answer:
            try:
                reachable = nx.shortest_path_length(sub.to_undirected(), center, answer) <= rec["hop_k"]
            except nx.NetworkXNoPath:
                pass
        out.append({
            "index": rec["index"], "image": rec["image"], "hop": rec["hop_k"],
            "nodes": sub.number_of_nodes(), "edges": sub.number_of_edges(),
            "nodes_list": sorted(sub.nodes()),
            "answer_in_subgraph": has_answer, "answer_reachable": reachable,
            "path_found": path_found,
        })
    return out


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n": n,
        "avg_nodes": sum(r["nodes"] for r in rows) / n,
        "avg_edges": sum(r["edges"] for r in rows) / n,
        "answer_in_subgraph": sum(r["answer_in_subgraph"] for r in rows) / n,
        "answer_reachable": sum(r["answer_reachable"] for r in rows) / n,
        "path_found": (sum(bool(r["path_found"]) for r in rows) / n
                       if any(r["path_found"] is not None for r in rows) else None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hop-filter", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    G, lookup = gen.load_data()
    records = load_records(args.hop_filter)

    conditions = {
        "S1-legacy": dict(strategy=1, directed=False, legacy=True),
        "S2-directed-legacy": dict(strategy=2, directed=True, legacy=True),
        "S2-undirected-legacy": dict(strategy=2, directed=False, legacy=True),
        "S1": dict(strategy=1, directed=False, legacy=False),
        "S2-undirected": dict(strategy=2, directed=False, legacy=False),
    }
    result = {}
    for name, cfg in conditions.items():
        rows = replay(G, lookup, records, seed=args.seed, **cfg)
        result[name] = {"summary": summarize(rows), "rows": rows}
        print(name, {k: (round(v, 3) if isinstance(v, float) else v)
                     for k, v in result[name]["summary"].items()})

    same = sum(a["nodes_list"] == b["nodes_list"] for a, b in
               zip(result["S1-legacy"]["rows"], result["S2-directed-legacy"]["rows"]))
    print(f"S1-legacy và S2-directed-legacy trùng tập node: {same}/{len(records)}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({k: v["summary"] for k, v in result.items()} |
                      {"rows": {k: v["rows"] for k, v in result.items()}},
                      f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
