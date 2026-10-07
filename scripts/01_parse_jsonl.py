"""
Bước 1: Lọc records MetaQA từ all_qa_data.jsonl
- Target: category "hop reasoning" (180 records) + "reasoning" movie domain (48 records)
- Extract entity name và hop count từ image filename
- Output: metaqa_subset.jsonl
"""

import json
import re
from pathlib import Path

JSONL_PATH = Path(__file__).parent.parent / "VKG-QA" / "all_qa_data.jsonl"
OUTPUT_PATH = Path(__file__).parent / "metaqa_subset.jsonl"

# Regex: "EntityName_2hop_subgraph.png" hoặc "EntityName_2hop_subgraph_iso.png"
SUBGRAPH_RE = re.compile(r"^(.+?)_(\d+)hop_subgraph(?:_iso)?\.png$")


def extract_entity_hop(image_filename: str) -> tuple[str, int] | None:
    m = SUBGRAPH_RE.match(image_filename)
    if not m:
        return None
    entity = m.group(1).strip()
    hop = int(m.group(2))
    return entity, hop


def is_metaqa_record(record: dict) -> bool:
    cat = record.get("category", "")
    if cat == "hop reasoning":
        return True
    if cat == "reasoning":
        domain = record.get("question domain", "") or record.get("question_domain", "")
        return str(domain).lower() == "movie"
    return False


def main():
    records = []
    skipped = 0

    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not is_metaqa_record(rec):
                continue

            result = extract_entity_hop(rec["image"])
            if result is None:
                skipped += 1
                continue

            entity, hop = result
            rec["entity_name"] = entity
            rec["hop_k"] = hop
            records.append(rec)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Thống kê
    categories = {}
    hops = {}
    entities = set()
    images = set()
    for r in records:
        categories[r["category"]] = categories.get(r["category"], 0) + 1
        hops[r["hop_k"]] = hops.get(r["hop_k"], 0) + 1
        entities.add(r["entity_name"])
        images.add(r["image"])

    print(f"Tổng records: {len(records)}")
    print(f"Skipped (không match pattern): {skipped}")
    print(f"Unique entities: {len(entities)}")
    print(f"Unique images cần sinh: {len(images)}")
    print(f"\nTheo category: {dict(sorted(categories.items()))}")
    print(f"Theo hop: {dict(sorted(hops.items()))}")
    print(f"\nOutput: {OUTPUT_PATH}")

    # In mẫu 5 entity đầu
    print("\nMẫu entities:")
    for e in sorted(entities)[:10]:
        print(f"  {e!r}")


if __name__ == "__main__":
    main()
