# VKG-QA — Tái lập thực nghiệm hop reasoning trên MetaQA

Mã nguồn tái lập một phần benchmark **VKG-QA** (Du et al., CVPR 2026): sinh ảnh đồ thị con từ MetaQA, hỏi mô hình đa phương thức qua Groq API theo kiểu zero-shot, chấm điểm và ghi nhận lên Weights & Biases.

Phần mô tả phương pháp và kết quả nằm trong báo cáo, không đặt ở đây.

## 1. Môi trường

- Python 3.12 (đã chạy trên Windows 10; Linux/macOS dùng lệnh tương đương)
- Không cần GPU: sinh ảnh chạy trên CPU, inference gọi API Groq
- Trình duyệt Chromium cho Playwright (dùng để chụp ảnh đồ thị)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

Tạo file `.env` từ `.env.example` và điền khóa Groq:

```
GROQ_API_KEY=...
```

## 2. Dữ liệu

| Dữ liệu | Nguồn | Đặt tại |
|---|---|---|
| Câu hỏi VKG-QA `all_qa_data.jsonl` | HuggingFace `sq413/VKG-QA` | `VKG-QA/all_qa_data.jsonl` |
| MetaQA knowledge base `kb.txt` | [MetaQA (Google Drive)](https://drive.google.com/drive/folders/0B-36Uca2AvwhTWVFSUZqRXVtbUE) | `MetaQA/kb/kb.txt` |
| Bộ chấm điểm chính thức | `https://github.com/jinchengyue123/vkgqa_eval` | `vkgqa_eval/` |

```bash
hf download sq413/VKG-QA all_qa_data.jsonl --repo-type dataset --local-dir VKG-QA
git clone https://github.com/jinchengyue123/vkgqa_eval.git
```

Repo đã kèm sẵn tập con đã lọc (`scripts/metaqa_subset.jsonl`), ảnh đã sinh (`scripts/images/`, `scripts/images_backup_strategy1/`) và kết quả inference (`scripts/results/`), nên có thể bỏ qua các bước tương ứng.

## 3. Chạy lại

Tất cả lệnh chạy trong thư mục `scripts/`.

| Bước | Lệnh | Đầu ra |
|---|---|---|
| 1. Lọc câu hỏi MetaQA | `python 01_parse_jsonl.py` | `metaqa_subset.jsonl` |
| 2. Dựng đồ thị tri thức | `python 02_build_kg.py` | `metaqa_kg.pkl` |
| 3a. Sinh ảnh, chiến lược 1 | `python 03_generate_images.py --strategy 1 --out-dir images_backup_strategy1` | ảnh PNG |
| 3b. Sinh ảnh, chiến lược 2 | `python 03_generate_images.py --strategy 2 --out-dir images` | ảnh PNG |
| 4. (tuỳ chọn) Kiểm tra model Groq | `python 05_check_models.py` | danh sách model vision |
| 5. Inference | `python 04_run_inference.py --model qwen/qwen3.8-27b [--hop-filter 2] [--output results/<tên>.jsonl]` | `results/*_results.jsonl` |
| 6. Chấm điểm | `python ../vkgqa_eval/eval.py --input results/<tên>_results.jsonl` | `*_evaluated.jsonl`, `*_summary.txt` |
| 7. Ghi nhận lên W&B | `wandb login` rồi `python 06_log_wandb.py --project vkgqa-reproduction` | các run trên W&B |

### Phân tích cắt bỏ (60 câu hop-2)

```bash
# Sinh ảnh cho từng cấu hình
python 03_generate_images.py --strategy 1 --hop-filter 2 --out-dir images_ablation/s1_fixed
python 03_generate_images.py --strategy 2 --hop-filter 2 --out-dir images_ablation/s2_fixed
python 03_generate_images.py --strategy 1 --hop-filter 2 --legacy-resolve --out-dir images_ablation/s1_labeled

# Inference + chấm điểm cho từng thư mục ảnh
python 04_run_inference.py --model qwen/qwen3.8-27b --hop-filter 2 --images-dir images_ablation/s1_fixed --output results/abl_s1_fixed_results.jsonl
python ../vkgqa_eval/eval.py --input results/abl_s1_fixed_results.jsonl

# Thống kê đồ thị con (số nút/cạnh, ảnh có chứa đáp án không) bằng cách phát lại bước trích
python 07_analyze_subgraphs.py --hop-filter 2 --out results/subgraph_stats_hop2.json
```

Các cờ của `03_generate_images.py`:

| Cờ | Ý nghĩa |
|---|---|
| `--strategy {1,2}` | 1 = BFS ngẫu nhiên; 2 = khóa đường suy luận tâm → đáp án |
| `--directed-path` | (chiến lược 2) chỉ tìm đường theo chiều cạnh |
| `--legacy-resolve` | dùng cách tra cứu thực thể cũ (trước khi sửa lỗi trùng tên khác hoa/thường) |
| `--hop-filter N` | chỉ sinh ảnh cho câu có hop = N |
| `--seed`, `--out-dir`, `--overwrite-hop` | seed, thư mục ra, sinh lại ảnh đã có |

Ghi chú:
- `03_generate_images.py` mặc định `--seed 42` và bỏ qua ảnh đã tồn tại; thêm `--overwrite-hop 2` để sinh lại ảnh có hop ≥ 2. Với cùng seed và cùng `--hop-filter`, tập nút/cạnh của đồ thị con lặp lại được giữa các lần chạy (bố cục ảnh do vis.js mô phỏng vật lý nên có thể khác).
- Khi gặp lỗi 429 (giới hạn token/ngày của Groq), `04_run_inference.py` tự chờ theo thời gian API gợi ý rồi chạy tiếp.
- `04_run_inference.py` đọc ảnh trong `scripts/images/`; thêm `--images-dir images_backup_strategy1` để chạy với ảnh chiến lược 1.
- `04_run_inference.py` tự nối tiếp (resume) nếu file output đã có một phần; lỗi API được ghi vào file `.log` cùng tên.
- `06_log_wandb.py --dry-run` in số liệu mà không cần tài khoản W&B. Danh sách run và file tương ứng khai báo trong biến `RUNS` của script.
- Model trên Groq có thể thay đổi theo thời gian; dùng bước 4 để chọn model vision còn khả dụng.
