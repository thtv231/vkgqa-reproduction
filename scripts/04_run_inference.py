"""
Bước 4: Chạy LMM inference qua Groq API (zero-shot, như paper VKG-QA).

Cách dùng:
  python 04_run_inference.py --model meta-llama/llama-4-maverick
  python 04_run_inference.py --model qwen/qwen3.6-27b
  python 04_run_inference.py --model qwen/qwen3.6-27b --dry-run 5

Output:
  results/{model_slug}_results.jsonl   (có thêm field "prediction")
  results/{model_slug}_errors.log      (record bị lỗi)

Sau đó eval:
  python ../vkgqa_eval/eval.py --input results/{model_slug}_results.jsonl
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# --- Config ---
SCRIPTS_DIR = Path(__file__).parent
load_dotenv(SCRIPTS_DIR.parent / ".env")

API_KEY = os.getenv("GROQ_API_KEY2") or os.getenv("GROQ_API_KEY")
if not API_KEY:
    print("[ERROR] Không tìm thấy GROQ_API_KEY / GROQ_API_KEY2 trong .env")
    sys.exit(1)

SUBSET_JSONL = SCRIPTS_DIR / "metaqa_subset.jsonl"
IMAGES_DIR   = SCRIPTS_DIR / "images"
RESULTS_DIR  = SCRIPTS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MAX_RETRIES    = 3
DELAY_BETWEEN  = 1.2   # giây giữa request (tránh rate limit)
MAX_TOKENS     = 512
TEMPERATURE    = 0     # deterministic — giống paper

# Groq dùng OpenAI-compatible endpoint
client = OpenAI(api_key=API_KEY, base_url="https://api.groq.com/openai/v1")


def model_to_slug(model_id: str) -> str:
    """Chuyển model ID thành tên file an toàn."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", model_id)


def image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def strip_thinking(text: str) -> str:
    """Xóa <think>...</think> block của Qwen3 CoT. Xử lý cả trường hợp chưa đóng tag."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)  # tag chưa đóng
    return cleaned.strip()


def call_groq(question: str, img_b64: str, model: str) -> str:
    """
    Zero-shot multimodal prompt — đúng format paper VKG-QA:
    image + question text, không có system prompt.
    Disable thinking mode cho Qwen3 để tiết kiệm tokens.
    """
    extra = {}

    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": question},
            ]
        }],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        **extra,
    )
    raw = resp.choices[0].message.content.strip()
    return strip_thinking(raw)   # strip <think> tags nếu còn sót


def load_existing(out_path: Path) -> set[int]:
    """Trả về set các index đã có trong output file (để resume)."""
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["index"])
                except Exception:
                    pass
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Groq model ID, vd: qwen/qwen3.6-27b")
    parser.add_argument("--dry-run", type=int, default=0,
                        help="Chỉ chạy N records đầu (0 = chạy tất cả)")
    parser.add_argument("--hop-filter", type=int, default=0,
                        help="Chỉ chạy records có hop == N (0 = tất cả)")
    parser.add_argument("--images-dir", default=str(IMAGES_DIR),
                        help="Thư mục ảnh subgraph (mặc định: scripts/images)")
    parser.add_argument("--output", default="",
                        help="Tên file output (mặc định: {slug}_results.jsonl)")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    model  = args.model
    slug   = model_to_slug(model)
    out_path = Path(args.output) if args.output else RESULTS_DIR / f"{slug}_results.jsonl"
    err_path = out_path.with_suffix(".log")

    # Đọc dataset
    records = []
    with open(SUBSET_JSONL, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    if args.hop_filter:
        records = [r for r in records if r.get("hop_k", r.get("hop", 0)) == args.hop_filter]
    if args.dry_run:
        records = records[:args.dry_run]

    total = len(records)
    done  = load_existing(out_path)
    todo  = [r for r in records if r["index"] not in done]

    print(f"Model  : {model}")
    print(f"Records: {total} total | {len(done)} đã có | {len(todo)} cần chạy")
    print(f"Output : {out_path}")
    if not todo:
        print("Tất cả records đã có. Dùng eval.py để đánh giá.")
        return

    ok = err = skip_img = 0

    with open(out_path, "a", encoding="utf-8") as fout, \
         open(err_path, "a", encoding="utf-8") as ferr:

        for i, rec in enumerate(todo, 1):
            img_path = images_dir / rec["image"]

            if not img_path.exists():
                skip_img += 1
                msg = f"[{i:3d}/{len(todo)}] SKIP — ảnh không có: {rec['image']}"
                print(msg); ferr.write(msg + "\n")
                continue

            pred = None
            for attempt in range(MAX_RETRIES):
                try:
                    img_b64 = image_to_base64(img_path)
                    pred = call_groq(rec["question"], img_b64, model)
                    break
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(2 ** attempt)
                    else:
                        msg = f"[{i:3d}/{len(todo)}] ERR idx={rec['index']}: {e}"
                        print(msg); ferr.write(msg + "\n")
                        err += 1

            if pred is not None:
                rec["prediction"] = pred
                rec["model"] = model
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"[{i:3d}/{len(todo)}] idx={rec['index']:4d} "
                      f"| {rec['category']:<14s} "
                      f"| ans={str(rec['answer'])[:20]!r:22s} "
                      f"| pred={pred[:40]!r}")
                ok += 1

            time.sleep(DELAY_BETWEEN)

    print(f"\nXong: {ok} OK | {err} lỗi | {skip_img} thiếu ảnh")
    print(f"\nChạy eval:")
    print(f"  cd scripts")
    print(f"  python ..\\vkgqa_eval\\eval.py \\")
    print(f"    --input results\\{slug}_results.jsonl \\")
    print(f"    --out-txt eval_reports\\{slug}_summary.txt")


if __name__ == "__main__":
    main()
