# Báo cáo đánh giá — AI Learning Assistant (RAG)

*Sinh tự động bởi `evaluation/evaluate.py` từ `evaluation/results.json` (lần chạy ngày 2026-11-05). Mọi con số trong bảng đến từ lần chạy này; mục 9 do người viết tay sau khi đọc kết quả.*

## 1. Tóm tắt

- Truy xuất (15 câu có đáp án): Hit@5 = 1.000, Recall@5 = 0.978, MRR = 1.000, Precision@3 = 0.778 (trần 1.000).
- Ngưỡng: hai lớp điểm top-1 **chồng lấn** (khoảng cách -0.0371); ngưỡng ít lỗi nhất trên lưới là 0.68 nhưng điểm top-1 gần nhất (q11) chỉ cách 0.0003; hiện tại 0.65 (mục 4.2).
- End-to-end: câu có đáp án đúng 12/15, đúng một phần 3, sai 0, bị chặn nhầm 0; câu không có đáp án xử lý đúng 5/5.
- Độ trễ tổng: median 1.6 s, p95 16.0 s, max 62.2 s.

## 2. Thiết lập

| Mục | Giá trị |
|---|---|
| Ngày chạy (UTC) | 2026-11-05 |
| Tài liệu | `postgresql_performance_guide_vi.pdf` — 9 trang, 29 chunk, doc_id `5d7d32354f75f17b` |
| Số câu hỏi | 20 |
| Mô hình embedding | `gemini-embedding-001`, 768 chiều |
| Mô hình trả lời | `gemini-3.1-flash-lite`, thinking = `minimal` |
| Mô hình chấm (judge) | `gemini-3.1-flash-lite`, thinking = `low`, temperature = 0 |
| Chunking | chunk_size = 800, chunk_overlap = 120 (ký tự) |
| Truy xuất | top_k = 5; K đánh giá = 3, 5 |
| Ngưỡng điểm (chạy end-to-end) | 0.65 (cosine, `SCORE_THRESHOLD` trong `.env`) |
| Các giai đoạn đã chạy | retrieval, end_to_end, judge |
| Lưu ý | `.env` đặt `gemini-3.5-flash` (thinking `minimal`); lần chạy này ghi đè bằng `--model`/`--thinking` cho riêng tiến trình này, `.env` không bị sửa |

## 3. Truy xuất (retrieval)

Đo trên 15 câu có đáp án trong tài liệu (single + multi). Một chunk được coi là đúng nếu **trang** của nó nằm trong `relevant_pages`. Precision@K = số chunk đúng trong K kết quả đầu / K. Recall@K = số trang đúng tìm thấy / số trang đúng. MRR tính trong top-5. Truy xuất chạy **không có ngưỡng điểm** để thấy điểm thật.

### 3.1 Tổng thể

| Chỉ số | K=3 | K=5 |
|---|---|---|
| Precision@K | 0.778 | 0.640 |
| Precision@K tối đa có thể đạt (trần) | 1.000 | 0.720 |
| Recall@K | 0.956 | 0.978 |
| Hit-rate@K | 1.000 | 1.000 |
| MRR (top-5) | 1.000 |  |

Lưu ý về *trần*: nhãn ở mức trang, mà mỗi trang chỉ có vài chunk; một câu có đáp án ở trang chỉ có 2 chunk thì Precision@5 không thể vượt 2/5 dù truy xuất hoàn hảo. Vì vậy hãy đọc Precision cùng với cột trần.

### 3.2 Theo loại câu hỏi

| Loại | n | P@3 | P@5 | R@3 | R@5 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|---|---|---|---|
| single (1 chunk) | 10 | 0.833 | 0.620 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| multi (nhiều chunk) | 5 | 0.667 | 0.680 | 0.867 | 0.933 | 1.000 | 1.000 | 1.000 |

### 3.3 Từng câu có đáp án

| Câu | Loại | Trang đúng | Top-5: trang (điểm) | P@3 | P@5 | R@5 | RR |
|---|---|---|---|---|---|---|---|
| q01 | single | 2 | 2 (0.794), 1 (0.703), 9 (0.677), 2 (0.674), 9 (0.665) | 0.333 | 0.400 | 1.000 | 1.000 |
| q02 | single | 2 | 2 (0.803), 2 (0.653), 2 (0.651), 1 (0.634), 1 (0.632) | 1.000 | 0.600 | 1.000 | 1.000 |
| q03 | single | 5 | 5 (0.836), 5 (0.733), 5 (0.715), 9 (0.667), 5 (0.652) | 1.000 | 0.800 | 1.000 | 1.000 |
| q04 | single | 5 | 5 (0.830), 5 (0.735), 5 (0.706), 6 (0.660), 6 (0.656) | 1.000 | 0.600 | 1.000 | 1.000 |
| q05 | single | 5 | 5 (0.752), 5 (0.703), 5 (0.676), 1 (0.652), 8 (0.642) | 1.000 | 0.600 | 1.000 | 1.000 |
| q06 | single | 6 | 6 (0.825), 6 (0.768), 4 (0.708), 6 (0.705), 6 (0.703) | 0.667 | 0.800 | 1.000 | 1.000 |
| q07 | single | 9 | 9 (0.759), 9 (0.707), 6 (0.695), 4 (0.673), 1 (0.650) | 0.667 | 0.400 | 1.000 | 1.000 |
| q08 | single | 3 | 3 (0.703), 4 (0.668), 3 (0.657), 4 (0.651), 3 (0.650) | 0.667 | 0.600 | 1.000 | 1.000 |
| q09 | single | 5 | 5 (0.787), 5 (0.729), 5 (0.699), 9 (0.659), 5 (0.656) | 1.000 | 0.800 | 1.000 | 1.000 |
| q10 | multi | 6 | 6 (0.781), 1 (0.658), 4 (0.645), 6 (0.627), 6 (0.606) | 0.333 | 0.600 | 1.000 | 1.000 |
| q11 | multi | 9 | 9 (0.680), 4 (0.677), 9 (0.665), 4 (0.664), 1 (0.655) | 0.667 | 0.400 | 1.000 | 1.000 |
| q12 | multi | 8 | 8 (0.749), 8 (0.724), 1 (0.686), 8 (0.681), 8 (0.662) | 0.667 | 0.800 | 1.000 | 1.000 |
| q13 | multi | 4 | 4 (0.776), 4 (0.666), 1 (0.658), 4 (0.644), 9 (0.639) | 0.667 | 0.600 | 1.000 | 1.000 |
| q14 | single | 7 | 7 (0.684), 7 (0.650), 7 (0.644), 2 (0.638), 6 (0.595) | 1.000 | 0.600 | 1.000 | 1.000 |
| q15 | multi | 4, 6, 8 | 6 (0.748), 6 (0.747), 6 (0.725), 6 (0.709), 4 (0.707) | 1.000 | 1.000 | 0.667 | 1.000 |

### 3.4 Câu không có đáp án (insufficient / unrelated): điểm top-1 thay cho Precision/Recall

| Câu | Loại | Câu hỏi | Điểm top-1 | Trang top-1 | Chunk top-1 |
|---|---|---|---|---|---|
| q16 | insufficient | pg_upgrade dùng để làm gì? | 0.6701 | 6 | 15 |
| q17 | insufficient | Cách triển khai sharding trong PostgreSQL? | 0.7174 | 1 | 1 |
| q18 | insufficient | So sánh hiệu năng PostgreSQL với MySQL? | 0.7123 | 1 | 1 |
| q19 | unrelated | Hướng dẫn nấu phở bò. | 0.5427 | 1 | 0 |
| q20 | unrelated | Ai vô địch World Cup gần nhất? | 0.4814 | 6 | 17 |

## 4. Nghiên cứu ngưỡng điểm

Dùng điểm top-1 của 20 câu. Một câu *qua cổng* khi có ít nhất một chunk đạt điểm ≥ ngưỡng, tức là khi top-1 ≥ ngưỡng.

| Nhóm | n | min | median | max |
|---|---|---|---|---|
| Có đáp án (single + multi) | 15 | 0.680 | 0.776 | 0.836 |
| Không có đáp án (insufficient + unrelated) | 5 | 0.481 | 0.670 | 0.717 |

- insufficient (thiếu bằng chứng): q16 = 0.6701, q17 = 0.7174, q18 = 0.7123
- unrelated (không liên quan): q19 = 0.5427, q20 = 0.4814

**Hai lớp CHỒNG LẤN**: câu không có đáp án cao nhất (q17) có top-1 = 0.7174, không thấp hơn câu có đáp án thấp nhất (q11) = 0.6803 (chênh -0.0371). Không có ngưỡng nào tách hoàn toàn hai lớp.

### 4.1 Bảng ngưỡng 0.55 → 0.80

| Ngưỡng | Chặn nhầm (/15) | Lọt qua nhầm (/5) | Tổng lỗi |
|---|---|---|---|
| 0.55 | 0 | 3 | 3 |
| 0.56 | 0 | 3 | 3 |
| 0.57 | 0 | 3 | 3 |
| 0.58 | 0 | 3 | 3 |
| 0.59 | 0 | 3 | 3 |
| 0.60 | 0 | 3 | 3 |
| 0.61 | 0 | 3 | 3 |
| 0.62 | 0 | 3 | 3 |
| 0.63 | 0 | 3 | 3 |
| 0.64 | 0 | 3 | 3 |
| 0.65 ← hiện tại | 0 | 3 | 3 |
| 0.66 | 0 | 3 | 3 |
| 0.67 | 0 | 3 | 3 |
| 0.68 | 0 | 2 | 2 |
| 0.69 | 2 | 2 | 4 |
| 0.70 | 2 | 2 | 4 |
| 0.71 | 3 | 2 | 5 |
| 0.72 | 3 | 0 | 3 |
| 0.73 | 3 | 0 | 3 |
| 0.74 | 3 | 0 | 3 |
| 0.75 | 5 | 0 | 5 |
| 0.76 | 7 | 0 | 7 |
| 0.77 | 7 | 0 | 7 |
| 0.78 | 8 | 0 | 8 |
| 0.79 | 10 | 0 | 10 |
| 0.80 | 11 | 0 | 11 |

*Chặn nhầm* = câu có đáp án nhưng top-1 < ngưỡng (người dùng nhận fallback dù tài liệu có đáp án). *Lọt qua nhầm* = câu không có đáp án nhưng top-1 ≥ ngưỡng (chunk được gửi cho LLM; LLM vẫn có thể từ chối).

### 4.2 Khuyến nghị

- Ngưỡng hiện tại 0.65: chặn nhầm 0, lọt qua nhầm 3.
- Ít lỗi nhất (2 lỗi): chỉ đúng một điểm của lưới, **0.68**.
- Ngưỡng cao nhất không chặn nhầm câu nào: 0.68 (khi đó 2/5 câu không có đáp án vẫn lọt qua).
- Biên an toàn của 0.68: điểm top-1 gần nhất là q11 = 0.6803 (cách 0.0003). Biên nhỏ hơn 0.01 nghĩa là chỉ cần câu hỏi diễn đạt khác đi một chút là kết quả đổi: đây là một điểm nằm sát mép, không phải một vùng an toàn.
- **Cảnh báo**: khuyến nghị được chọn trên chính 20 câu này (và chỉ một tài liệu) nên lạc quan. Nó cho biết dải hợp lý, không phải con số đã được kiểm chứng trên dữ liệu mới.

### 4.3 Thử ý tưởng hai ngưỡng (chỉ đo, không đổi `rag.py`)

Cổng (*gate*) = top-1 phải đạt mức này thì mới trả lời; *include* = chỉ chunk có điểm ≥ include mới vào prompt (include ≤ gate; include = gate là cách làm hiện tại). *Precision/Recall chunk gửi đi* tính ở mức trang trên các câu đã qua cổng (precision) và trên mọi câu có đáp án, câu bị chặn tính 0 (recall).

| Gate | Include | Chặn nhầm | Lọt qua nhầm | Chunk gửi/câu | Precision chunk gửi | Recall chunk gửi | Chunk nhiễu/câu lọt |
|---|---|---|---|---|---|---|---|
| 0.65 | 0.65 | 0 | 3 | 4.067 | 0.701 | 0.978 | 4.333 |
| 0.65 | 0.63 | 0 | 3 | 4.800 | 0.632 | 0.978 | 5.000 |
| 0.65 | 0.60 | 0 | 3 | 4.933 | 0.650 | 0.978 | 5.000 |
| 0.65 | 0.55 | 0 | 3 | 5.000 | 0.640 | 0.978 | 5.000 |
| 0.68 | 0.68 | 0 | 2 | 2.400 | 0.914 | 0.978 | 4.000 |
| 0.68 | 0.66 | 0 | 2 | 3.200 | 0.817 | 0.978 | 5.000 |
| 0.68 | 0.63 | 0 | 2 | 4.800 | 0.632 | 0.978 | 5.000 |
| 0.68 | 0.58 | 0 | 2 | 5.000 | 0.640 | 0.978 | 5.000 |

Giới hạn: top-5 là trần của số chunk gửi; recall chỉ tính trong 5 chunk đã truy xuất.

## 5. End-to-end (RagPipeline.answer)

Ngưỡng 0.65, top_k = 5, không lọc tài liệu.

### 5.1 Câu có đáp án

| Kết quả | Số câu (/15) |
|---|---|
| Đúng | 12 |
| Đúng một phần | 3 |
| Sai | 0 |
| Bị ngưỡng chặn nhầm (fallback) | 0 |
| Chưa chấm / thiếu | 0 |

Tỉ lệ đúng nghiêm ngặt (chỉ 'Đúng', câu bị chặn tính là sai): 12/15. Đúng hoặc đúng một phần: 15/15.

### 5.2 Câu không có đáp án (đúng = fallback, hoặc LLM từ chối đúng)

| Kết quả | Số câu (/5) |
|---|---|
| Fallback đúng (không gọi LLM) | 2 |
| LLM được gọi và từ chối đúng | 3 |
| Trả lời nhầm như thể tài liệu có | 0 |
| Chưa chấm / thiếu | 0 |

Xử lý đúng: 5/5.

### 5.3 Trung thực (faithfulness) và trích dẫn

- Judge đã chấm 18 câu có gọi LLM. Trung thực (không có khẳng định nào thiếu căn cứ trong chunk): **18/18**; riêng các câu không phải từ chối: 15/15.
- Câu trả lời có ít nhất một trích dẫn hợp lệ [n]: 17/18.
- Câu có đáp án mà ít nhất một nguồn được trích nằm ở trang đúng: 15/15.
- Số trích dẫn tới nguồn không tồn tại (đã bị loại khỏi câu trả lời): 0.

### 5.4 Từng câu

| Câu | Loại | Kết quả | Fallback | Top-1 | Trang gửi LLM | Trang được trích | Trung thực | Tổng (s) | Retry |
|---|---|---|---|---|---|---|---|---|---|
| q01 | single | Đúng | không | 0.794 | 2, 1, 9, 2, 9 | 2 | có | 1.5 | 0 |
| q02 | single | Đúng | không | 0.803 | 2, 2, 2 | 2 | có | 1.5 | 0 |
| q03 | single | Đúng | không | 0.836 | 5, 5, 5, 9, 5 | 5 | có | 2.1 | 0 |
| q04 | single | Đúng | không | 0.830 | 5, 5, 5, 6, 6 | 5 | có | 62.2 | 1 |
| q05 | single | Đúng | không | 0.752 | 5, 5, 5, 1 | 5 | có | 1.5 | 0 |
| q06 | single | Đúng | không | 0.825 | 6, 6, 4, 6, 6 | 6 | có | 1.3 | 0 |
| q07 | single | Đúng | không | 0.759 | 9, 9, 6, 4 | 9 | có | 2.2 | 0 |
| q08 | single | Đúng | không | 0.703 | 3, 4, 3, 4 | 3 | có | 1.6 | 0 |
| q09 | single | Đúng | không | 0.787 | 5, 5, 5, 9, 5 | 5 | có | 13.6 | 0 |
| q10 | multi | Đúng | không | 0.781 | 6, 1 | 6 | có | 1.6 | 0 |
| q11 | multi | Đúng | không | 0.680 | 9, 4, 9, 4, 1 | 9 | có | 1.7 | 0 |
| q12 | multi | Đúng một phần | không | 0.749 | 8, 8, 1, 8, 8 | 8, 8, 8 | có | 1.7 | 0 |
| q13 | multi | Đúng một phần | không | 0.776 | 4, 4, 1 | 4, 4 | có | 2.0 | 0 |
| q14 | single | Đúng | không | 0.684 | 7 | 7 | có | 1.5 | 0 |
| q15 | multi | Đúng một phần | không | 0.748 | 6, 6, 6, 6, 4 | 6, 4 | có | 1.9 | 0 |
| q16 | insufficient | LLM từ chối đúng | không | 0.670 | 6, 1, 8 | – | có | 1.4 | 0 |
| q17 | insufficient | LLM từ chối đúng | không | 0.717 | 1, 7, 7, 1, 2 | 7 | có | 1.3 | 0 |
| q18 | insufficient | LLM từ chối đúng | không | 0.712 | 1, 1, 2, 4, 9 | 1 | có | 1.5 | 0 |
| q19 | unrelated | Fallback đúng | có | 0.543 | – | – | – | 0.3 | 0 |
| q20 | unrelated | Fallback đúng | có | 0.481 | – | – | – | 0.3 | 0 |

## 6. Độ trễ

| Nhóm (total_ms, giây) | Median | p95 | Max |
|---|---|---|---|
| Tất cả (20 câu) | 1.6 | 16.0 | 62.2 |
| Chỉ câu có gọi LLM (18 câu) | 1.6 | 20.9 | 62.2 |

p95 tính bằng nội suy tuyến tính giữa hai hạng gần nhất (như NumPy). Câu chậm nhất: q04. Số câu có retry: 1 (tổng 1 lần retry; thời gian chờ backoff nằm trong total_ms).
Câu fallback (không gọi LLM) có median 0.3 s, chủ yếu là một lần embed câu hỏi.

Đuôi chậm của LLM free tier là vấn đề đã biết (KI-18 trong `docs/known_issues.md`): với N nhỏ, p95 gần như chỉ là câu chậm nhất, nên đừng đọc nó như một con số ổn định.

## 7. Kiểm tra độ tin cậy của judge

Judge là một LLM nên **không phải sự thật tuyệt đối**: có thể quá dễ dãi, thiên vị câu dài, hoặc thiên vị văn phong của chính mô hình (ở đây judge và mô hình trả lời cùng một mô hình). Hai cách kiểm tra bên dưới đều chưa thay được việc người thật đọc lại.

### 7.1 5 verdict chọn ngẫu nhiên để bạn tự kiểm tra (seed 20261105)

**q04** — maintenance_work_mem mặc định là bao nhiêu và dùng cho tác vụ nào?

- Đáp án chuẩn: Mặc định 64MB; dùng cho các tác vụ bảo trì như VACUUM và CREATE INDEX.
- Câu trả lời của hệ thống: `maintenance_work_mem` có giá trị mặc định là 64MB và được sử dụng cho các tác vụ bảo trì như VACUUM và CREATE INDEX [1].
- Judge: **correct**, từ chối = False, trung thực = True. The system answer correctly identifies the default value (64MB) and the usage (maintenance tasks like VACUUM and CREATE INDEX) as stated in the reference.

**q05** — PgBouncer có những chế độ pooling nào?

- Đáp án chuẩn: Ba chế độ: session pooling, transaction pooling và statement pooling.
- Câu trả lời của hệ thống: PgBouncer có ba chế độ pooling: session pooling, transaction pooling và statement pooling [1].
- Judge: **correct**, từ chối = False, trung thực = True. The system answer correctly identifies all three pooling modes mentioned in the reference answer.

**q07** — View pg_stat_activity dùng để làm gì?

- Đáp án chuẩn: Cho biết các truy vấn đang chạy và các phiên 'idle in transaction'.
- Câu trả lời của hệ thống: View `pg_stat_activity` được dùng để cho biết các truy vấn đang chạy và các phiên ở trạng thái 'idle in transaction' [1].
- Judge: **correct**, từ chối = False, trung thực = True. The system answer accurately reflects the information provided in the reference answer and the retrieved chunks.

**q10** — MVCC là gì và vì sao nó dẫn đến hiện tượng phình bảng?

- Đáp án chuẩn: MVCC: thao tác đọc không chặn thao tác ghi và ngược lại. Mỗi UPDATE tạo phiên bản dòng mới, bản cũ thành dòng chết; nếu không được dọn thì bảng và chỉ mục phình to và truy vấn chậm dần.
- Câu trả lời của hệ thống: MVCC (Multi-Version Concurrency Control) là cơ chế của PostgreSQL cho phép các thao tác đọc không chặn thao tác ghi và ngược lại [1].  MVCC dẫn đến hiện tượng phình bảng vì mỗi lệnh UPDATE không sửa đổi dữ liệu tại chỗ mà tạo ra một phiên bản dòng mới, khiến phiên bản cũ trở thành "dòng chết" (dead tuple) [1]. Nếu các dòng chết này không được dọn dẹp, bảng và chỉ mục sẽ dần trở nên phình to (bloat) [1].
- Judge: **correct**, từ chối = False, trung thực = True. The system answer correctly defines MVCC and explains the mechanism of row versioning and dead tuples leading to table bloat, matching all key points in the reference answer.

**q18** — So sánh hiệu năng PostgreSQL với MySQL?

- Đáp án chuẩn: Tài liệu không đề cập MySQL.
- Câu trả lời của hệ thống: Tài liệu được cung cấp không chứa thông tin so sánh hiệu năng giữa PostgreSQL và MySQL. Tài liệu này chỉ tập trung vào các kỹ thuật chẩn đoán và cải thiện hiệu năng dành riêng cho hệ quản trị cơ sở dữ liệu PostgreSQL [2].
- Judge: **correct**, từ chối = True, trung thực = True. The system correctly identifies that the document does not contain information about MySQL and explains that the document focuses on PostgreSQL performance optimization.

### 7.2 Ý kiến thứ hai: Claude (Sonnet 5.5) — một LLM khác họ với judge, không phải con người

Tôi đọc từng câu trả lời, đối chiếu với đáp án chuẩn trong questions.json và với NỘI DUNG CHUNK thật đã gửi cho LLM (không dùng kiến thức ngoài). Đây là ý kiến thứ hai của một LLM, không phải sự thật tuyệt đối; người dùng nên tự đọc lại các câu ở mục 7.1 và các câu ranh giới.

Trên 18 câu judge đã chấm: đồng ý về mức đúng/sai 18/18, đồng ý về trung thực 18/18.

Không có câu nào bất đồng về nhãn. Điều đó chưa chứng minh judge đúng: hai LLM có thể cùng mắc một kiểu sai.

Các câu mà ý kiến thứ hai coi là **ranh giới** (nhãn có thể đổi tùy cách đọc tiêu chí):

- **q11** (judge: correct; ý kiến thứ hai: correct): Đủ nguyên nhân, cách khắc phục và 3 cái giá (6 phút, 1,4GB, INSERT chậm 4%) đúng chunk 27, nhưng bỏ số p95 giảm từ 2,8 s còn 12 ms vốn nằm trong mệnh đề 'khắc phục' của đáp án chuẩn. Câu hỏi không hỏi kết quả đo nên tôi chấm đúng; theo tiêu chí 'mọi ý chính' nghiêm ngặt thì có thể là một phần.

## 8. Câu sai hoặc đáng ngờ (phát hiện tự động từ số đo)

- **q04**: chậm: 62.2 s (retry: 1)
- **q09**: chậm: 13.6 s (retry: 0)
- **q12**: judge: partially_correct — The system answer covers most key points but misses 'replication slot' which is explicitly mentioned in the reference answer and the retrieved chunks.
- **q13**: judge: partially_correct — The system answer lists 6 errors, but misses 'thống kê lỗi thời' and 'transaction mở quá lâu' mentioned in the reference answer. It also includes an extra point 'So sánh hai cột khác kiểu dữ liệu' which is part of the first point in the source text, not a separate error.
- **q15**: Recall@5 = 2/3 trang đúng; judge: partially_correct — The system answer correctly identifies long-running transactions, idle in transaction, and replication slots as factors. It misses the 'hot_standby_feedback' factor mentioned in the reference answer.
- **q16**: vượt ngưỡng dù tài liệu không có đáp án (top-1 = 0.6701); kết quả: LLM từ chối đúng
- **q17**: vượt ngưỡng dù tài liệu không có đáp án (top-1 = 0.7174); kết quả: LLM từ chối đúng
- **q18**: vượt ngưỡng dù tài liệu không có đáp án (top-1 = 0.7123); kết quả: LLM từ chối đúng

## 9. Phân tích nguyên nhân và đề xuất cải thiện

*Phần này do người viết tay sau khi đọc `results.json` và nội dung 29 chunk gốc. Các con số chép từ lần chạy ngày 2026-11-05 (mô hình trả lời `gemini-3.1-flash-lite`, thinking `minimal`); nếu chạy lại **không** dùng cache, phần liên quan đến LLM có thể lệch nhẹ. Nhận xét của judge (mục 7) do chính judge viết bằng tiếng Anh.*

### 9.1 Bối cảnh lần chạy (đọc trước khi tin các con số)

- `gemini-3.5-flash` (mô hình trong `.env`) **hết quota ngày** khi chạy: lần thử đầu, cả 18 câu cần LLM đều lỗi `rate_limited` (server yêu cầu chờ 14991 s ≈ 4,2 giờ), hai câu fallback vẫn chạy được. (Một lời gọi thăm dò ngắn ngay trước đó còn qua được, nên quota gần như cạn đúng lúc chạy.) Script không điền gì thay thế; sau đó chạy lại với `gemini-3.1-flash-lite` + thinking `minimal` theo chỉ dẫn. Hệ quả: **chất lượng câu trả lời của 3.5-flash chưa được đo**, và judge cùng họ (cùng mô hình) với mô hình trả lời nên có nguy cơ tự ưu ái.
- Một số câu trong bộ test trùng với bộ dùng để đặt ngưỡng tạm thời 0.65 (KI-17): PgBouncer (q05), `pg_upgrade` (q16), sharding (q17), nấu phở (q19). Vì vậy ngưỡng 0.65 chưa được kiểm tra hoàn toàn độc lập.

### 9.2 Kết luận chính

- **Tìm đúng trang thì tốt**: Hit@5 = 1.000 và MRR = 1.000 (trang đúng luôn ở hạng 1 với cả 15 câu). Điểm yếu nằm ở **chọn chunk** và **độ đầy đủ của câu trả lời**, không phải ở việc tìm trang.
- Không có câu nào trả lời sai hẳn, bị chặn nhầm, hay bị judge/tôi đánh giá là bịa (18/18 trung thực). Ba câu "đúng một phần" (q12, q13, q15) đều là câu **liệt kê thiếu mục**.
- Năm câu không có đáp án được xử lý đúng 5/5, nhưng chỉ **2/5 nhờ ngưỡng** (q19, q20); 3/5 (q16, q17, q18) vượt ngưỡng và được **LLM từ chối đúng**.
- Judge và tôi không bất đồng nhãn nào trên 18 câu, chỉ có một câu ranh giới (q11). Điều này không chứng minh judge đúng (cả hai đều là LLM), nên 5 verdict ngẫu nhiên ở mục 7.1 vẫn cần bạn tự đọc.

### 9.3 Nguyên nhân gốc của các câu sai hoặc đáng ngờ

Ba câu "đúng một phần" có **ba nguyên nhân khác nhau**; tôi kiểm tra bằng cách xếp hạng cả 29 chunk cho từng câu (vector đã cache, 3 lần embed câu hỏi).

| Câu | Triệu chứng | Nguyên nhân gốc (đã kiểm) | Bằng chứng |
|---|---|---|---|
| q12 | Thiếu *replication slot* | **Sinh câu trả lời**: chunk 24 (có "Replication slot giúp...") xếp hạng 5, điểm 0.6615 ≥ 0.65 nên **đã nằm trong prompt** (nguồn [5]) nhưng LLM không dùng. Mô hình lite với thinking `minimal` bỏ sót ý ở chunk cuối. | Hạng 5/29, 0.6615; nguồn [5] không được trích |
| q13 | Thiếu 2/7 lỗi (thống kê lỗi thời, transaction mở quá lâu) | **Chọn chunk bằng ngưỡng**: hai ý này nằm ở chunk 10, xếp hạng 4 với điểm 0.6437 < 0.65 nên bị loại. Trong khi đó chunk mục lục (chunk 1, 0.6577) lại *lọt qua* và chiếm một chỗ trong prompt. | Hạng 4/29, 0.6437; Recall mức trang vẫn = 1.0 vì chunk 8, 9 cùng trang 4 |
| q15 | Thiếu `hot_standby_feedback` | **Truy xuất**: chunk 23 (trang 8) chỉ xếp hạng 11 (0.5923) vì chỉ là ý phụ trong đoạn nói về replica; bốn chunk trang 6 chiếm top-4. Tăng `top_k` từ 5 lên 10 cũng không đủ (cần ≥ 11 và ngưỡng ≤ 0.59, sẽ kéo thêm nhiều nhiễu). Nhãn `relevant_pages = [4, 6, 8]` cũng khá khắt khe vì đáp án chuẩn ghi đây là ý "ngoài ra". | Hạng 11/29, 0.5923; Recall@5 = 2/3 trang |

Các câu đáng ngờ khác:

- **q16, q17, q18 (insufficient)**: cả ba vượt ngưỡng 0.65 (0.6701, 0.7174, 0.7123), và cả ba được LLM từ chối đúng. An toàn ở đây đến từ **LLM + prompt**, không phải từ ngưỡng; N = 3 và chỉ một mô hình nên chưa thể coi là bảo đảm. Đáng chú ý: ở KI-17, câu `pg_upgrade dùng để làm gì và chạy như thế nào?` đạt 0.6493 (bị fallback với biên 0.0007), còn câu ngắn hơn `pg_upgrade dùng để làm gì?` đạt **0.6701** và vượt ngưỡng. Chỉ đổi cách diễn đạt đã dịch điểm 0.021, đủ để đổi kết quả: ngưỡng 0.65 không ổn định trước cách hỏi.
- **Vì sao hai lớp chồng lấn?** Không chỉ do chunk bìa/mục lục (KI-14): chunk 1 (mục lục) là top-1 của q17 và q18 (0.717, 0.712) và có mặt trong 9/18 prompt. Nhưng bỏ chunk 0 và 1 khỏi xét thì q17 vẫn đạt 0.7144 (chunk 21 nói về phân vùng, gần nghĩa với "sharding") trong khi câu có đáp án thấp nhất là q11 = 0.6803. Đây là kiểu "gần chủ đề nhưng không có đáp án" mà một ngưỡng cosine không tách được.
- **q11**: top-1 chỉ 0.6803, thấp nhất trong các câu có đáp án (chỉ cao hơn ngưỡng 0.65 đúng 0.03). Câu trả lời bỏ số p95 (2,8 s → 12 ms) nên là ca ranh giới đúng/một phần.
- **q04 (62.2 s) và q09 (13.6 s)**: 2/18 lần gọi LLM chậm hơn 10 s, trong khi median của `llm_ms` là 1.23 s. Ở q04, `llm_ms` = 61.8 s và có 1 retry sau `HTTP 500 INTERNAL` (backoff 1.1 s); suy ra lần gọi đầu mất khoảng 60 s, trùng `timeout_s = 60` của `GeminiLLM` (suy luận từ số đo, không có log chi tiết từng pha). Đây đúng là đuôi chậm của KI-18, vẫn chưa được xử lý.
- **Precision@5 = 0.640 trông thấp nhưng không phải lỗi**: trần đạt được chỉ là 0.720 vì nhiều trang chỉ có 3 chunk.
- **KI-01 / KI-14**: 9/18 prompt chứa một chunk bìa/mục lục (trang 1) và chưa thấy chúng gây sai hay bịa trong lần chạy này, nhưng chúng chiếm chỗ trong ngữ cảnh và làm nhiễu top-1 của câu không có đáp án.

### 9.4 Nên làm gì với ngưỡng (kết luận của tôi, dựa trên N = 20 và một tài liệu)

- **Không nên đổi sang 0.68** dù đó là điểm lưới ít lỗi nhất (2 lỗi): nó là một điểm đơn lẻ và q11 chỉ vượt 0.68 đúng 0.0003; chỉ ở 0.69 đã có 2 câu có đáp án bị chặn nhầm.
- **Giữ 0.65 cho tới khi có thêm dữ liệu**: 0 câu bị chặn nhầm, 3 câu lọt qua đều bị LLM từ chối đúng. Tôi ưu tiên không chặn nhầm vì chặn nhầm làm người dùng mất câu trả lời có thật, còn lọt qua còn có lưới an toàn thứ hai (prompt). Đây là phán đoán về sự đánh đổi, chưa được kiểm chứng ngoài bộ câu hỏi này.
- **Ý tưởng hai ngưỡng**, đo ở mức trang: hạ ngưỡng chọn chunk từ 0.65 xuống 0.55 giữ nguyên Recall (0.978) nhưng tăng số chunk gửi từ 4.07 lên 5.0 và giảm precision chunk gửi từ 0.701 xuống 0.640, tức **không thấy lợi ích ở mức trang**. Tuy vậy q13 cho thấy lợi ích có thể có ở mức chunk (hạ ngưỡng chọn xuống 0.63 sẽ đưa chunk 10 vào) mà nhãn mức trang không đo được. Chưa đủ bằng chứng để đổi `rag.py`.

### 9.5 Ba đề xuất cải thiện cụ thể

1. **Nhãn bằng chứng mức chunk và bộ câu hỏi lớn hơn, tách dev/test.** Thêm vào `questions.json` danh sách chunk (hoặc đoạn văn) bắt buộc phải có trong ngữ cảnh, thêm ít nhất 2 tài liệu khác nhau và tối thiểu ~50 câu; chỉnh ngưỡng trên tập dev, báo cáo trên tập test chưa từng nhìn. Lý do: q13 cho thấy nhãn trang che mất lỗi thật, và ngưỡng đang được chọn trên chính bộ test. Kiểm tra: Recall mức chunk của q13 phải lộ ra (< 1.0) trước khi sửa.
2. **Cải thiện bước chọn chunk, rồi đo lại bằng đúng bộ này.** Loại chunk bìa/mục lục khỏi chỉ mục hoặc khỏi phép chọn (KI-14, KI-01); thử *ngưỡng cổng* trên top-1 và *ngưỡng chọn* thấp hơn, hoặc cắt theo điểm tương đối so với top-1; đánh giá ở mức chunk. Kỳ vọng cần kiểm tra: q13 lấy lại được chunk 10 mà không làm tăng số câu lọt qua của q16–q18; nếu không đạt thì bỏ ý tưởng.
3. **Giảm đuôi chậm và sót mục của bước sinh.** (a) Đặt thời gian chờ cứng cho cả lần gọi LLM (ví dụ ~20 s, tối đa 1 retry) thay vì chỉ `timeout_s` theo từng pha (KI-18); mục tiêu là không còn câu nào > 20 s như q04. (b) Với câu dạng liệt kê (q12, q13, q15), thử `thinking = low` hoặc thêm một dòng vào prompt "liệt kê đủ mọi mục có trong các chunk" và chạy lại bộ này với `gemini-3.5-flash` khi có quota; kiểm tra bằng số câu "đúng một phần" và độ trễ median.

### 9.6 Điều chưa biết

- Chất lượng thật của `gemini-3.5-flash` trên bộ này (hết quota).
- Judge chỉ được đối chiếu với ý kiến của một LLM khác (tôi), chưa có người chấm độc lập.
- Kết quả trên tài liệu dài, nhiều tài liệu, DOCX/TXT (không có số trang) hoàn toàn chưa được đo.

## 10. Hạn chế

- **Một tài liệu duy nhất, 9 trang, 29 chunk**: kết quả không nói gì về tài liệu dài, nhiều tài liệu cùng lúc, hay tài liệu có bảng/hình.
- **N nhỏ (20 câu, 15 có đáp án)**: mỗi câu đổi chiều làm tỉ lệ đổi 5–7 điểm phần trăm; không có khoảng tin cậy nào đáng tin ở cỡ này. Đây là kiểm tra hồi quy và phát hiện lỗi, không phải benchmark.
- **Nhãn ở mức trang**: một chunk ở đúng trang nhưng không chứa đáp án vẫn được tính là đúng; ngược lại đáp án nằm ở nhiều trang thì nhãn có thể thiếu. Precision@K bị chặn trên bởi số chunk của trang (xem cột trần).
- **Judge thiên lệch**: LLM chấm LLM; xem mục 7. Nhãn tham chiếu do chủ dự án tự viết từ PDF, cũng có thể sai sót.
- **Ngưỡng được chọn và đánh giá trên cùng một bộ câu hỏi**; ngoài ra một số câu (PgBouncer, `pg_upgrade`, sharding, nấu phở) trùng với các câu đã dùng khi đặt ngưỡng tạm 0.65 (KI-17), nên ngưỡng đó không hoàn toàn độc lập với bộ test.
- **LLM không tất định** (temperature 0.2) và free tier có đuôi chậm: chạy lại không cache có thể cho kết quả khác một chút. Cache trong `evaluation/cache/` giữ nguyên lần chạy này.
- Không đo: ảnh hưởng của header/footer lặp lại (KI-01), chunk bìa/mục lục (KI-14) một cách riêng biệt — chỉ quan sát được qua danh sách chunk truy xuất ở mục 3.3.

## 11. Cách chạy lại

```powershell
docker compose up -d                                   # Qdrant
.venv\Scripts\python.exe evaluation/evaluate.py        # chạy đủ 4 giai đoạn (dùng cache nếu có)
.venv\Scripts\python.exe evaluation/evaluate.py --skip-llm      # chỉ truy xuất + nghiên cứu ngưỡng
.venv\Scripts\python.exe evaluation/evaluate.py --report-only   # dựng lại report.md từ results.json
.venv\Scripts\python.exe -m pytest tests/test_eval_metrics.py tests/test_eval_judge.py -q
```

Mọi lời gọi API được cache trong `evaluation/cache/` theo hash của đầu vào, nên chạy lại không tốn quota và cho đúng các số này; xóa thư mục đó để chạy lại từ đầu. Kết quả thô nằm ở `evaluation/results.json`.
