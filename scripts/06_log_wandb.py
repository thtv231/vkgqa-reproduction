"""
Bước 6: Ghi nhận kết quả inference đã chấm (results/*_evaluated.jsonl) lên Weights & Biases.

Không gọi lại API — chỉ đọc field "correct" do vkgqa_eval/eval.py sinh ra.

Cách dùng:
  python 06_log_wandb.py --project vkgqa-reproduction
  python 06_log_wandb.py --dry-run          # chỉ in số liệu, không log
"""

import argparse
import json
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPTS_DIR / "results"

# Mỗi run ↔ một file đã chấm. image_dir: thư mục ảnh tương ứng với lần chạy đó.
RUNS = [
    {
        "name": "S1-qwen3.6-27b-hop1",
        "file": "qwen_qwen3_6-27b_results_evaluated.jsonl",
        "group": "preliminary",
        "strategy": "S1 (BFS ngẫu nhiên)",
        "image_dir": "images_backup_strategy1",
    },
    {
        "name": "S1-qwen3.8-27b",
        "file": "qwen_qwen3_8-27b_results_evaluated.jsonl",
        "group": "main-results",
        "strategy": "S1 (BFS ngẫu nhiên)",
        "image_dir": "images_backup_strategy1",
    },
    {
        "name": "S2v1-qwen3.8-27b",
        "file": "qwen_qwen3_8-27b_s2_results_evaluated.jsonl",
        "group": "ablation-subgraph-strategy",
        "strategy": "S2 v1 (khóa đường suy luận)",
        "image_dir": None,   # ảnh S2 v1 đã bị ghi đè bởi v2
    },
    {
        "name": "S2v2-qwen3.8-27b-hop2",
        "file": "qwen_s2_hop2_v2_results_evaluated.jsonl",
        "group": "main-results",
        "strategy": "S2 v2 (khóa đường suy luận)",
        "image_dir": "images",
    },
]

N_IMAGE_SAMPLES = 10


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def accuracy(rows: list[dict]) -> float | None:
    return sum(r["correct"] for r in rows) / len(rows) if rows else None


def summarize(rows: list[dict]) -> dict:
    out = {"n_samples": len(rows), "acc_overall": accuracy(rows)}
    for h in (1, 2, 3):
        sub = [r for r in rows if r["hop_k"] == h]
        if sub:
            out[f"n_hop{h}"] = len(sub)
            out[f"acc_hop{h}"] = accuracy(sub)
    return out


def shared_hop2(all_rows: dict[str, list[dict]]) -> tuple[set[int], dict[str, float]]:
    """Tập index hop-2 có mặt ở mọi run có hop-2 → so sánh công bằng."""
    hop2 = {name: {r["index"]: r for r in rows if r["hop_k"] == 2}
            for name, rows in all_rows.items()}
    hop2 = {k: v for k, v in hop2.items() if v}
    common = set.intersection(*(set(v) for v in hop2.values()))
    acc = {name: sum(v[i]["correct"] for i in common) / len(common) for name, v in hop2.items()}
    return common, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="vkgqa-reproduction")
    parser.add_argument("--entity", default=None, help="W&B entity (user/team), mặc định là tài khoản đăng nhập")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    all_rows = {run["name"]: load(RESULTS_DIR / run["file"]) for run in RUNS}
    common, common_acc = shared_hop2(all_rows)

    for run in RUNS:
        print(run["name"], summarize(all_rows[run["name"]]))
    print(f"Hop-2 trên {len(common)} câu chung:", {k: round(v, 4) for k, v in common_acc.items()})
    if args.dry_run:
        return

    import wandb

    for run in RUNS:
        rows = all_rows[run["name"]]
        model = rows[0].get("model", "")
        config = {
            "model": model,
            "provider": "Groq (OpenAI-compatible API)",
            "strategy": run["strategy"],
            "source_file": run["file"],
            "temperature": 0,
            "max_tokens": 512,
            "prompt": "zero-shot, image + question, no system prompt",
            "dataset": "VKG-QA hop reasoning (MetaQA subset)",
        }
        wb = wandb.init(project=args.project, entity=args.entity, name=run["name"],
                        group=run["group"], job_type="eval", config=config, reinit=True)

        metrics = summarize(rows)
        if run["name"] in common_acc:
            metrics["acc_hop2_shared"] = common_acc[run["name"]]
            metrics["n_hop2_shared"] = len(common)
        wb.summary.update(metrics)

        table = wandb.Table(columns=["index", "hop", "entity", "question", "answer",
                                     "prediction", "correct", "image"])
        n_img = 0
        for r in rows:
            img = None
            if run["image_dir"] and n_img < N_IMAGE_SAMPLES:
                p = SCRIPTS_DIR / run["image_dir"] / r["image"]
                if p.exists():
                    img = wandb.Image(str(p), caption=r["image"])
                    n_img += 1
            table.add_data(r["index"], r["hop_k"], r["entity_name"], r["question"],
                           str(r["answer"]), r["prediction"], r["correct"], img)
        wb.log({"predictions": table})
        wb.finish()

    # Run tổng hợp: so sánh S1 vs S2 trên cùng tập câu hop-2
    wb = wandb.init(project=args.project, entity=args.entity, name="compare-hop2-shared",
                    group="main-results", job_type="analysis", reinit=True,
                    config={"n_hop2_shared": len(common)})
    cmp = wandb.Table(columns=["run", "acc_hop2_shared"],
                      data=[[k, v] for k, v in common_acc.items()])
    wb.log({"hop2_shared_comparison": wandb.plot.bar(cmp, "run", "acc_hop2_shared",
                                                     title=f"Hop-2 accuracy ({len(common)} câu chung)")})
    wb.finish()


if __name__ == "__main__":
    main()
