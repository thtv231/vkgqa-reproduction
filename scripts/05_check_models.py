"""
Bước 0: Kiểm tra Groq models nào available và hỗ trợ vision.
Chạy trước khi inference để biết dùng model nào.
"""

import base64
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / ".env")

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    print("[ERROR] Không tìm thấy GROQ_API_KEY trong .env")
    sys.exit(1)

client = OpenAI(api_key=API_KEY, base_url="https://api.groq.com/openai/v1")

# --- Bước 1: Lấy danh sách model ---
print("=" * 60)
print("Danh sách models trên Groq:")
print("=" * 60)

models = client.models.list()
model_ids = sorted(m.id for m in models.data)
for mid in model_ids:
    print(f"  {mid}")

# --- Bước 2: Lọc vision candidates ---
VISION_KEYWORDS = ["llava", "vision", "vl", "qwen", "llama-4", "maverick", "scout", "gemma"]
candidates = [m for m in model_ids
              if any(kw in m.lower() for kw in VISION_KEYWORDS)]

print(f"\nVision candidates ({len(candidates)}):")
for c in candidates:
    print(f"  → {c}")

# --- Bước 3: Test vision với ảnh mẫu ---
SAMPLE_IMAGE = Path(__file__).parent / "images" / "The Shawshank Redemption_1hop_subgraph.png"

if not SAMPLE_IMAGE.exists():
    print("\n[SKIP] Không tìm thấy ảnh mẫu để test vision")
    sys.exit(0)

with open(SAMPLE_IMAGE, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

TEST_QUESTION = "How many nodes are there in this graph? Just give the number."

print(f"\n{'='*60}")
print("Test vision với ảnh: The Shawshank Redemption_1hop_subgraph.png")
print(f"Question: {TEST_QUESTION}")
print("=" * 60)

working_models = []
for model_id in candidates:
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": TEST_QUESTION}
                ]
            }],
            max_tokens=64,
            temperature=0,
        )
        answer = resp.choices[0].message.content.strip()
        print(f"  [OK] {model_id}")
        print(f"       → {answer!r}")
        working_models.append(model_id)
    except Exception as e:
        err = str(e)[:120]
        print(f"  [FAIL] {model_id}: {err}")

print(f"\n{'='*60}")
print(f"Models hỗ trợ vision: {len(working_models)}/{len(candidates)}")
for m in working_models:
    print(f"  ✓ {m}")

if working_models:
    print(f"\nGợi ý dùng cho inference:")
    print(f"  python 04_run_inference.py --model {working_models[0]}")
