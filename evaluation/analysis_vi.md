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
