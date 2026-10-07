"""
Bước 3: Sinh ảnh VKG từ MetaQA subgraph bằng pyvis + Playwright.

Pipeline per record:
  1. Tìm central entity trong MetaQA KG
  2. BFS k-hop → subgraph (có edge constraint |E| <= 1.2*|V|)
  3. Render bằng pyvis → HTML
  4. Screenshot bằng Playwright → PNG

Cách dùng:
  python 03_generate_images.py --strategy 1 --out-dir images_backup_strategy1
  python 03_generate_images.py --strategy 2 --overwrite-hop 2
"""

import argparse
import json
import pickle
import random
import re
import tempfile
from pathlib import Path

import networkx as nx
from pyvis.network import Network

# --- Paths ---
SCRIPTS_DIR = Path(__file__).parent
SUBSET_JSONL = SCRIPTS_DIR / "metaqa_subset.jsonl"
GRAPH_PKL = SCRIPTS_DIR / "metaqa_kg.pkl"
IMAGES_DIR = SCRIPTS_DIR / "images"

# --- Node màu theo hop distance (phỏng đoán từ paper) ---
HOP_COLORS = {
    0: "#dd4b39",   # central entity → đỏ
    1: "#00ff1e",   # hop-1 → xanh lá
    2: "#162347",   # hop-2 → xanh navy
    3: "#f5a905",   # hop-3 → vàng cam
}
HOP_SIZES = {0: 28, 1: 18, 2: 14, 3: 11}

# Edge constraint từ paper: |E| <= w * |V|
EDGE_WEIGHT = {1: 1.2, 2: 1.2, 3: 1.5}


def load_data():
    with open(GRAPH_PKL, "rb") as f:
        data = pickle.load(f)
    return data["graph"], data["lookup"]


def resolve_entity(entity_name: str, lookup: dict) -> str | None:
    """Tìm canonical entity name, thử exact rồi lowercase."""
    if entity_name in lookup.values():
        return entity_name
    lower = entity_name.lower()
    return lookup.get(lower)


MAX_NODES = 25   # giới hạn nodes để graph vừa render được
MAX_NBR_PER_NODE = 5   # sample tối đa 5 neighbors mỗi node mỗi hop



def extract_subgraph(G: nx.MultiDiGraph, center: str, k: int) -> tuple[nx.MultiDiGraph, dict]:
    """
    BFS k-hop từ center với hard cap MAX_NODES.
    Mỗi node expand tối đa MAX_NBR_PER_NODE neighbors (random sample).
    Áp dụng edge constraint |E| <= w*|V| sau BFS.
    """
    visited = {center: 0}
    frontier = [center]

    for hop in range(1, k + 1):
        next_frontier = []
        for node in frontier:
            if len(visited) >= MAX_NODES:
                break
            neighbors = sorted(set(G.successors(node)) | set(G.predecessors(node)))
            # Ưu tiên successors (outgoing edges) trước
            successors = [n for n in neighbors if n in G.successors(node) and n not in visited]
            others = [n for n in neighbors if n not in G.successors(node) and n not in visited]
            # Sample để không quá dày
            random.shuffle(successors)
            random.shuffle(others)
            candidates = (successors + others)[:MAX_NBR_PER_NODE]
            for nbr in candidates:
                if nbr not in visited and len(visited) < MAX_NODES:
                    visited[nbr] = hop
                    next_frontier.append(nbr)
        frontier = next_frontier

    sub = G.subgraph(list(visited.keys())).copy()

    # Apply edge constraint: |E| <= w * |V|
    w = EDGE_WEIGHT.get(k, 1.5)
    max_edges = int(w * sub.number_of_nodes())
    if sub.number_of_edges() > max_edges:
        all_edges = list(sub.edges(keys=True, data=True))
        center_edges = [(u, v, key, d) for u, v, key, d in all_edges
                        if u == center or v == center]
        other_edges = [(u, v, key, d) for u, v, key, d in all_edges
                       if u != center and v != center]
        random.shuffle(other_edges)
        keep = center_edges + other_edges[:max(0, max_edges - len(center_edges))]
        sub = nx.MultiDiGraph()
        for u, v, _, d in keep:
            sub.add_edge(u, v, **d)
        if center not in sub:
            sub.add_node(center)

    return sub, visited


def find_reasoning_path(G: nx.MultiDiGraph, start: str, end: str, max_k: int = 3):
    """
    Tìm đường đi ngắn nhất start→end trên đồ thị VÔ HƯỚNG (bidirectional).
    MetaQA hop-2 thường cần đi ngược chiều cạnh, ví dụ:
      Film →directed_by→ Director ←directed_by← OtherFilm
    Trả về (path_nodes, path_edges) hoặc None nếu không tìm thấy / path quá dài.
    """
    try:
        G_undir = G.to_undirected()
        path_nodes = nx.shortest_path(G_undir, start, end)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    if len(path_nodes) - 1 > max_k:
        return None
    # Tái tạo directed edges theo chiều có trong G gốc
    path_edges = []
    for u, v in zip(path_nodes, path_nodes[1:]):
        if G.has_edge(u, v):
            edges = dict(G[u][v])
            key = next(iter(edges))
            path_edges.append((u, v, key, edges[key]))
        elif G.has_edge(v, u):
            edges = dict(G[v][u])
            key = next(iter(edges))
            path_edges.append((v, u, key, edges[key]))
    return path_nodes, path_edges


def extract_subgraph_strategy2(G: nx.MultiDiGraph, center: str, answer: str, k: int):
    """
    Strategy 2: Lock reasoning path center→answer, rồi BFS mở rộng ngữ cảnh.
    Đảm bảo answer entity luôn có mặt trong subgraph.
    Fallback sang Strategy 1 nếu không tìm được path.
    """
    result = find_reasoning_path(G, center, answer, max_k=k)
    if result is None:
        return extract_subgraph(G, center, k)

    path_nodes, path_edges = result
    locked_edges = set((u, v, key) for u, v, key, _ in path_edges)

    # Bước 1: Lock reasoning path, gán hop distance theo vị trí
    visited = {}
    for i, node in enumerate(path_nodes):
        visited[node] = min(i, k)

    # Bước 2: BFS contextual expansion từ tất cả locked nodes
    frontier = list(visited.keys())
    for hop in range(1, k + 1):
        if len(visited) >= MAX_NODES:
            break
        next_frontier = []
        for node in frontier:
            if len(visited) >= MAX_NODES:
                break
            successors = [n for n in G.successors(node) if n not in visited]
            others = [n for n in G.predecessors(node) if n not in visited and n not in successors]
            random.shuffle(successors)
            random.shuffle(others)
            candidates = (successors + others)[:MAX_NBR_PER_NODE]
            for nbr in candidates:
                if nbr not in visited and len(visited) < MAX_NODES:
                    visited[nbr] = hop
                    next_frontier.append(nbr)
        frontier = next_frontier

    sub = G.subgraph(list(visited.keys())).copy()

    # Bước 3: Edge constraint — giữ locked path edges, trim phần còn lại
    w = EDGE_WEIGHT.get(k, 1.5)
    max_edges = int(w * sub.number_of_nodes())
    if sub.number_of_edges() > max_edges:
        all_edges = list(sub.edges(keys=True, data=True))
        kept = [(u, v, key, d) for u, v, key, d in all_edges if (u, v, key) in locked_edges]
        other_edges = [(u, v, key, d) for u, v, key, d in all_edges if (u, v, key) not in locked_edges]
        random.shuffle(other_edges)
        kept += other_edges[:max(0, max_edges - len(kept))]
        sub = nx.MultiDiGraph()
        for u, v, _, d in kept:
            sub.add_edge(u, v, **d)
        if center not in sub:
            sub.add_node(center)

    return sub, visited


def build_pyvis(sub: nx.MultiDiGraph, hop_dist: dict, center: str) -> Network:
    net = Network(
        height="1000px",
        width="1600px",
        directed=True,
        bgcolor="#ffffff",
        font_color="#000000",
        notebook=False,
        cdn_resources="in_line",
    )
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=180)

    for node in sub.nodes():
        d = hop_dist.get(node, 3)
        color = HOP_COLORS.get(d, HOP_COLORS[3])
        size = HOP_SIZES.get(d, HOP_SIZES[3]) * 1.4
        label = node if len(node) <= 22 else node[:19] + "..."
        net.add_node(node, label=label, color=color, size=size,
                     shape="dot", title=node,
                     font={"size": 16, "color": "#111111",
                           "strokeWidth": 3, "strokeColor": "#ffffff"})

    for u, v, data in sub.edges(data=True):
        rel = data.get("relation", "")
        net.add_edge(u, v, title=rel, label=rel,
                     font={"size": 13, "color": "#111111",
                           "strokeWidth": 3, "strokeColor": "#ffffff",
                           "align": "middle"},
                     arrows={"to": {"enabled": True, "scaleFactor": 0.7}},
                     color={"color": "#888888"},
                     width=1.5)

    return net


def render_png(html_path: Path, out_path: Path):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.goto(f"file:///{html_path.as_posix()}", timeout=60000)
        page.wait_for_timeout(3000)   # chờ physics settle (ms)
        page.screenshot(path=str(out_path), full_page=False)
        browser.close()


def process_record(G, lookup, rec, strategy: int, images_dir: Path, overwrite_hop: int) -> str:
    entity = rec["entity_name"]
    hop_k = rec["hop_k"]
    image_filename = rec["image"]
    out_path = images_dir / image_filename

    if out_path.exists():
        if not (overwrite_hop and hop_k >= overwrite_hop):
            return f"[SKIP] {image_filename} đã tồn tại"

    canonical = resolve_entity(entity, lookup)
    if canonical is None:
        return f"[MISS] Entity '{entity}' không tìm thấy trong MetaQA KG"

    answer_raw = rec.get("answer", "")
    # answer có thể dạng "A|B|C" (multiple), lấy đáp án đầu tiên
    answer_first = answer_raw.split("|")[0].strip() if answer_raw else ""
    answer_canonical = resolve_entity(answer_first, lookup) if answer_first else None

    if strategy == 2 and answer_canonical and rec.get("category", "") == "hop reasoning":
        sub, hop_dist = extract_subgraph_strategy2(G, canonical, answer_canonical, hop_k)
    else:
        sub, hop_dist = extract_subgraph(G, canonical, hop_k)
    net = build_pyvis(sub, hop_dist, canonical)

    html_content = net.generate_html()
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as tf:
        html_path = Path(tf.name)
        tf.write(html_content)

    try:
        render_png(html_path, out_path)
    finally:
        html_path.unlink(missing_ok=True)

    return f"[OK]   {image_filename} ({sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=int, choices=[1, 2], default=2,
                        help="1 = BFS ngẫu nhiên; 2 = khóa đường suy luận (cho hop reasoning)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=str(IMAGES_DIR),
                        help="Thư mục lưu ảnh PNG")
    parser.add_argument("--overwrite-hop", type=int, default=0,
                        help="Sinh lại ảnh đã có với hop >= N (0 = không ghi đè)")
    args = parser.parse_args()
    random.seed(args.seed)
    images_dir = Path(args.out_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    if not GRAPH_PKL.exists():
        print("[ERROR] metaqa_kg.pkl chưa có. Chạy 02_build_kg.py trước.")
        return
    if not SUBSET_JSONL.exists():
        print("[ERROR] metaqa_subset.jsonl chưa có. Chạy 01_parse_jsonl.py trước.")
        return

    G, lookup = load_data()

    # Đọc records, deduplicate theo image filename
    seen_images = {}
    with open(SUBSET_JSONL, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            img = rec["image"]
            if img not in seen_images:
                seen_images[img] = rec

    records = list(seen_images.values())
    print(f"Cần sinh {len(records)} ảnh unique...\n")

    ok = miss = skip = 0
    for i, rec in enumerate(records, 1):
        msg = process_record(G, lookup, rec, args.strategy, images_dir, args.overwrite_hop)
        print(f"[{i:3d}/{len(records)}] {msg}")
        if msg.startswith("[OK]"):
            ok += 1
        elif msg.startswith("[MISS]"):
            miss += 1
        else:
            skip += 1

    print(f"\nKết quả: {ok} sinh mới | {skip} bỏ qua (đã có) | {miss} không tìm thấy entity")
    print(f"Ảnh lưu tại: {images_dir}")


if __name__ == "__main__":
    main()
