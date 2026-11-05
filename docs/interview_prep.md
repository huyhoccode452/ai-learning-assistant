# Tài liệu chuẩn bị phỏng vấn — AI Learning Assistant

Tài liệu này dựa trên code thật, `README.md`, `evaluation/report.md` và `docs/known_issues.md`. Mọi con số lấy từ lần chạy
đánh giá ngày 2026-11-05 (1 tài liệu PDF 9 trang, 29 chunk, 20 câu hỏi). Nếu bạn chạy lại và số khác, **hãy nói số bạn thấy**,
đừng đọc thuộc lòng số trong file này.

> Nguyên tắc khi trả lời: nêu kết quả, nêu cách đo, rồi nêu giới hạn. Người phỏng vấn đánh giá cao ứng viên tự chỉ ra điểm yếu
> của mình trước khi bị hỏi. Mã `KI-xx` là mục tương ứng trong `docs/known_issues.md`.

---

## 1. Kịch bản demo 3 phút

### Chuẩn bị trước (5 phút, làm trước buổi phỏng vấn)

1. Bật **Docker Desktop**, rồi `docker compose up -d`. (Máy bạn từng tắt Docker Desktop nên Qdrant không chạy, và 29 test cần Qdrant bị skip.)
2. Hai terminal: `uvicorn app.main:app --reload` và `streamlit run frontend/streamlit_app.py` (chạy từ thư mục gốc dự án).
3. Mở http://127.0.0.1:8501, **tải sẵn** `data/samples/postgresql_performance_guide_vi.pdf` và chờ trạng thái sẵn sàng.
   Nếu để lúc demo mới tải, đo thật là khoảng 3 s khi quota mới, nhưng free tier có thể chậm hơn (KI-12).
4. Chạy thử **một lần** mỗi thứ để chắc quota còn: một câu hỏi đáp, một tóm tắt ngắn, một quiz 5 câu. Kết quả tóm tắt và quiz được lưu
   trong SQLite nên lần lặp lại tốn 0 lượt gọi LLM (KI-28). Đây vừa là phương án dự phòng, vừa là điểm để nói về cache.
5. Mở sẵn `evaluation/report.md` và thư mục `docs/screenshots/` trong một tab khác (dùng khi cần dự phòng, xem mục 1.2).

### 1.1 Kịch bản (tổng ~3 phút)

| Thời gian | Thao tác trên UI | Nên nói |
|---|---|---|
| 0:00–0:15 | Chỉ vào thanh bên: tài liệu đã `sẵn sàng`. Chọn tài liệu. | "Đây là ứng dụng hỏi đáp trên tài liệu của riêng mình. PDF được trích chữ giữ số trang, cắt chunk, embed rồi lưu vào Qdrant. Việc này chạy nền nên giao diện không bị treo." |
| 0:15–0:55 | Tab **Hỏi đáp**. Gõ câu có đáp án trong **một chunk** (q03): `work_mem mặc định là bao nhiêu và dùng cho thao tác nào?` Mở **Nguồn (n)**. | "Câu trả lời có trích dẫn [n]. Mở Nguồn thì thấy file, số trang, điểm tương đồng và đoạn trích. Mã kiểm tra trích dẫn: số nào không tồn tại sẽ bị xóa khỏi câu trả lời." Kết quả mong đợi: đáp án 4MB, nguồn ở trang 5 (lần chạy đánh giá: top-1 = 0.836, 2.1 s). |
| 0:55–1:25 | Câu **nhiều chunk** (q10): `MVCC là gì và vì sao nó dẫn đến hiện tượng phình bảng?` | "Câu này cần ghép định nghĩa MVCC với cơ chế dòng chết, mô hình tổng hợp từ nhiều đoạn và vẫn gắn trích dẫn." (Tránh q13 và q15 trong demo: hai câu này đã được chấm "đúng một phần" vì liệt kê thiếu mục.) |
| 1:25–2:00 | Câu **liên quan nhưng tài liệu không có** (q18): `So sánh hiệu năng PostgreSQL với MySQL?` | "Đây là ca khó. Câu hỏi cùng chủ đề nên điểm tương đồng vẫn vượt ngưỡng (0.7123 ≥ 0.65), một ngưỡng không chặn được. Lúc này lớp an toàn thứ hai là prompt: mô hình phải nói tài liệu không đề cập MySQL." Kết quả mong đợi: LLM từ chối. Đây **không phải** màn hình fallback vì LLM đã được gọi. Mô hình không tất định, nên nếu câu chữ khác chút thì không sao, miễn là nó không bịa. |
| 2:00–2:20 | Câu **không liên quan** (q19): `Hướng dẫn nấu phở bò.` | "Câu này top-1 chỉ 0.5427, dưới ngưỡng nên hệ thống trả lời 'không đủ bằng chứng' và **không gọi LLM**: tốn 0 quota, không thể bịa, và phản hồi nhanh (0.3 s trong lần đánh giá)." Chỉ vào dòng điểm số ở thông báo. |
| 2:20–2:40 | Chuyển sang tab **Tóm tắt**, chọn *Ngắn gọn*, bấm **Tạo tóm tắt**. | "Đọc toàn bộ chunk của tài liệu từ Qdrant, gửi một lần gọi LLM khi vừa 60.000 ký tự, nếu dài hơn thì map-reduce có giới hạn cứng. Kết quả được lưu nên lần sau 0 lượt gọi." |
| 2:40–3:00 | Tab **Trắc nghiệm**, số câu 5, giữ ô kiểm tra lại được chọn, bấm **Tạo bộ câu hỏi**. Làm một câu rồi bấm **Kiểm tra**. | "Mỗi câu có đoạn trích dẫn nguyên văn. Code kiểm tra đoạn đó thật sự nằm trong chunk được nêu, và code tự xáo đáp án. Nó chứng minh đáp án có trong tài liệu, chứ không chứng minh câu hỏi hay (KI-28)." |

Nếu còn thời gian: mở thanh bên **Lịch sử** để cho thấy hội thoại được lưu và mở lại **không gọi LLM**, rồi nói thẳng rằng
câu hỏi tiếp theo **không** dùng ngữ cảnh các tin nhắn trước (KI-29).

### 1.2 Nếu demo lỗi

| Tình huống | Dấu hiệu | Cách xử lý |
|---|---|---|
| **Hết quota Gemini** (429, `RATE_LIMITED`) | Câu hỏi hiện thông báo lỗi quota. Ở lần đánh giá, `gemini-3.5-flash` hết quota ngày (KI-18). | (1) Nói ngay: "Đây là giới hạn free tier, hệ thống báo lỗi nhanh chứ không treo hàng giờ." (2) Vẫn demo được: **câu không liên quan** (không gọi LLM), tóm tắt/quiz **đã lưu**, và mở **Lịch sử** (không gọi LLM). Câu hỏi mới cần LLM thì không chạy được. (3) Mở `docs/screenshots/07-error-429.png` và `answer-with-sources.png`. (4) Hạn mức tính theo từng mô hình (KI-18), nên có thể đổi `GEMINI_LLM_MODEL` trong `.env` sang mô hình khác còn quota rồi khởi động lại API. Việc này **không** cần index lại, vì chỉ đổi embedding mới phải index lại. Chỉ làm nếu bạn đã kiểm chứng tên mô hình từ trước. |
| **Qdrant tắt** (Docker Desktop không chạy) | `GET /health` trả `degraded` (`qdrant_reachable: false`). Upload và hỏi đáp không chạy được. | `docker compose up -d` (nhớ mở Docker Desktop trước). Dữ liệu vector nằm trong volume `qdrant_data` nên còn nguyên sau khi khởi động lại. Nếu volume bị xóa mà SQLite vẫn ghi `ready` thì xóa tài liệu và tải lại (KI-25). Trong lúc chờ, mở `docs/screenshots/08-backend-down.png` để cho thấy giao diện xử lý lỗi và nói về xử lý lỗi. Phần UI khi *riêng Qdrant* tắt tôi **chưa** kiểm tra bằng trình duyệt thật, nên đừng hứa trước màn hình cụ thể. |
| **API (uvicorn) chưa chạy** | Giao diện báo không kết nối được, có nút **Thử kết nối lại**. | Khởi động `uvicorn`, bấm nút. |
| **Tài liệu kẹt ở `processing`** | Free tier embed chậm (~100 đoạn mỗi phút, KI-12). | Dùng tài liệu đã tải sẵn từ bước chuẩn bị. Giao diện tự kiểm tra mỗi 2 s và bỏ cuộc sau 180 s, khi đó bấm **Kiểm tra lại**. |
| **Câu trả lời chậm bất thường** | Đuôi chậm của free tier: lần đánh giá có câu mất 62.2 s (KI-18). | Nói: "median 1.6 s, nhưng có đuôi chậm do server; cải thiện cần timeout tổng cứng." Chuyển sang bước khác trong lúc chờ. |
| **Câu trả lời khác với tài liệu này** | LLM không tất định (temperature 0.2). | Đừng cãi với màn hình. Mở nguồn, chỉ ra đoạn trích có/không có. Nói: "đây là lý do tôi đo bằng bộ câu hỏi thay vì tin vào một lần chạy." |

---

## 2. Pitch 30 giây

> "Em xây một ứng dụng hỏi đáp trên tài liệu cá nhân, dùng RAG. Người dùng tải PDF, DOCX hoặc TXT lên, đặt câu hỏi và nhận câu trả lời có
> trích dẫn đến đúng trang. Điểm em chú trọng là **trung thực**: nếu tài liệu không có thông tin, hệ thống nói 'không đủ bằng chứng',
> và với câu hoàn toàn không liên quan thì **không gọi LLM**. Toàn bộ pipeline em tự viết, không dùng LangChain, vì em muốn giải thích được
> từng bước. Em đo bằng bộ 20 câu hỏi: trên một tài liệu 9 trang, Hit@5 và MRR đều đạt 1.000, 12/15 câu có đáp án đúng hoàn toàn,
> 3 câu đúng một phần, không câu nào sai. Em cũng ghi rõ giới hạn: bộ đánh giá nhỏ, chỉ một tài liệu, và ngưỡng điểm chưa vững."

---

## 3. 20 câu hỏi phỏng vấn và câu trả lời ngắn

**1. Vì sao chọn Gemini embedding thay vì BGE-M3?**
Máy chỉ có CPU, không GPU, và deadline 2 ngày. BGE-M3 nặng khoảng 2 GB và chậm trên CPU, còn API Gemini miễn phí, nhanh và xử lý được tiếng Việt. Em để embedding sau interface `Embedder` (Protocol trong `app/embeddings.py`) nên có thể đổi sang BGE-M3 mà không sửa pipeline. Đánh đổi: phụ thuộc mạng và rate limit (~100 đoạn mỗi phút, KI-12); và đổi mô hình hoặc số chiều thì phải tạo lại collection. Em dùng `gemini-embedding-001`, 768 chiều, cấu hình trong `.env`. *(README, `app/embeddings.py`)*

**2. Vì sao không dùng LangChain?**
Đây là dự án để học và phải bảo vệ được từng phần, nên em tự viết từng bước: trích xuất, làm sạch, chia chunk, embed, tìm kiếm, lọc ngưỡng, dựng prompt, kiểm tra trích dẫn. Ít phụ thuộc hơn, dễ debug hơn và giải thích được. Ngoại lệ duy nhất là `langchain-text-splitters` cho `RecursiveCharacterTextSplitter`, đã ghim phiên bản 1.1.3. Cái giá: kéo theo `langchain-core` và khoảng 13 gói khác, và có một lỗi upstream (với regex separator và `keep_separator=False` thì mã regex lọt vào text chunk). Em tránh nó bằng `keep_separator=True` và có test canh giữ (KI-11). *(`app/ingestion.py`, KI-11)*

**3. Quy trình chunking và vì sao 800/120?**
Mỗi trang được cắt **riêng**, nên một chunk không bao giờ nằm giữa hai trang và số trang trích dẫn luôn đúng. Splitter đệ quy thử lần lượt các dấu phân cách: đoạn văn, dòng, câu, mệnh đề, từ, rồi ký tự. Chunk ngắn hơn 30 ký tự bị bỏ (số trang lẻ, tiêu đề cô đơn). **Về 800/120 em nói thật: đó là giá trị khởi đầu, em chưa chạy thí nghiệm so sánh nhiều kích thước.** Lý do hợp lý: 800 ký tự vừa đủ cho một ý hoàn chỉnh, thấp xa giới hạn 2048 token của embedding, và overlap 120 (15%) giảm nguy cơ cắt đôi một ý. Kết quả đo được trên bộ này là Hit@5 = 1.000, nhưng điều đó không chứng minh 800/120 là tối ưu. Hạn chế đã biết: tiêu đề có thể rơi sang chunk khác với phần thân (KI-02). *(`app/ingestion.py`, `app/config.py`)*

**4. Chunk ID idempotent hoạt động thế nào?**
`doc_id` là 16 ký tự hex đầu của SHA-256 nội dung file. `chunk_id = uuid5(namespace cố định, f"{doc_id}:{chunk_index}")`. Cùng file thì cùng id, nên tải lại ghi đè chứ không nhân đôi vector. Trước khi upsert, `QdrantStore.index_document` xóa mọi điểm của `doc_id` đó, nhờ vậy khi cấu hình chunk đổi và số chunk giảm thì không còn chunk cũ sót lại (KI-08, có test). Hai lưu ý: namespace không bao giờ được đổi sau khi đã index; và xóa-rồi-ghi không atomic (KI-13), nhưng an toàn khi gọi lại. *(`app/ingestion.py`, `app/vectorstore.py`)*

**5. Chọn ngưỡng 0.65 thế nào và vì sao không hoàn hảo?**
Ban đầu em đo điểm top-1 của vài nhóm câu hỏi rồi đặt tạm 0.65. Sau đó chạy nghiên cứu ngưỡng trên 20 câu (mục 4 của report): câu có đáp án có top-1 từ 0.680 đến 0.836, câu không có đáp án từ 0.481 đến 0.717. **Hai lớp chồng lấn** 0.037 (q17 = 0.7174 cao hơn q11 = 0.6803). Câu không liên quan tách rất rõ (≤ 0.543), nhưng câu *liên quan mà tài liệu không có* thì vượt ngưỡng: ở 0.65 cả 3 câu q16–q18 lọt qua và LLM từ chối đúng cả 3. Cùng một chủ đề, đổi cách diễn đạt đã làm `pg_upgrade` dịch từ 0.6493 lên 0.6701, vượt ngưỡng. Một ngưỡng cosine không tách được loại câu này. Em ưu tiên *không chặn nhầm* câu có đáp án (0 chặn nhầm ở 0.65). *(`app/rag.py`, KI-17, report mục 4)*

**6. Các lớp chống hallucination là gì?**
Sáu lớp, mỗi lớp có giới hạn: (1) ngưỡng điểm, nếu không chunk nào đạt thì trả "không đủ bằng chứng" và không gọi LLM; (2) prompt hệ thống: chỉ dùng ngữ cảnh, không bịa số liệu/tên/trích dẫn, nói rõ khi thiếu, và kiến thức chung phải nằm trong đoạn riêng có nhãn; (3) bắt buộc trích dẫn `[n]`; (4) code loại bỏ số trích dẫn không tồn tại (`resolve_citations`); (5) lọc theo tài liệu và chỉ tìm trong tài liệu `ready`; (6) đo bằng đánh giá: judge chấm trung thực 18/18. Giới hạn: lớp (4) chỉ chứng minh số tồn tại chứ không chứng minh chunk hỗ trợ câu đó (KI-20), và lớp (6) do LLM chấm (KI-26). *(`app/rag.py`, `app/prompts.py`)*

**7. Precision@K, Recall@K, MRR là gì, và vì sao trần Precision@5 là 0.72?**
Nhãn của em ở mức **trang** (`relevant_pages`). Precision@K = số chunk đúng trong K kết quả đầu / K. Recall@K = số trang đúng tìm được / số trang đúng. MRR = trung bình của 1/hạng của kết quả đúng đầu tiên: 1.000 nghĩa là với cả 15 câu, kết quả đầu tiên đã đúng. Trần: nhiều trang trong tài liệu chỉ có vài chunk, mà trang đó là nơi chứa đáp án. Nếu trang chỉ có 3 chunk thì dù truy xuất hoàn hảo, Precision@5 cũng không vượt 3/5 = 0.6. Trung bình trên 15 câu, mức tối đa đạt được là 0.720 (ở K=3 là 1.000), nên P@5 = 0.640 thực chất là khoảng 89% của mức trần, chứ không phải "thấp". Vì vậy phải đọc Precision cùng cột trần. *(report mục 3.1, 3.2)*

**8. Giới hạn của LLM-as-judge là gì? (KI-26)**
Judge là một LLM nên không phải sự thật tuyệt đối. Ở đây judge và mô hình trả lời là **cùng một mô hình** (`gemini-3.1-flash-lite`), nên có nguy cơ tự ưu ái. Em kiểm tra bằng ý kiến thứ hai từ một LLM khác họ (Claude): đồng ý 18/18 về đúng/sai và 18/18 về trung thực, chỉ đánh dấu q11 là ca ranh giới. Nhưng hai LLM đồng ý thì chưa phải bằng chứng; cần một người đọc lại, mà em chưa có người chấm độc lập. Các giới hạn khác của đánh giá: chỉ 20 câu (15 có đáp án), một tài liệu, nhãn mức trang, và ngưỡng được tinh chỉnh trên chính 4 câu có trong bộ test (q05, q16, q17, q19). *(report mục 7, 10, KI-26)*

**9. Vì sao việc index chạy nền?**
Embedding trên free tier giới hạn khoảng 100 đoạn mỗi phút. Đo thật: ba batch 100 đoạn mất 85 s, 70 s, 78 s (có cả retry 429). Một tài liệu 500 chunk sẽ mất vài phút, không thể giữ một request HTTP mở. Nên `POST /documents` kiểm tra và trích xuất nhanh rồi trả `202`, một `BackgroundTask` embed và upsert, client poll `GET /documents/{doc_id}` (`processing` → `ready`/`failed`). Hạn chế: `BackgroundTasks` chạy trong tiến trình, tắt server là mất job (khi khởi động lại các dòng `processing` được đánh dấu `failed` với mã `interrupted`), không có hàng đợi (KI-22). Giải pháp thật sự là task queue. *(`app/main.py`, KI-12, KI-22)*

**10. Prompt injection được xử lý thế nào?**
Chunk là dữ liệu do tác giả tài liệu viết, có thể chứa "bỏ qua mọi hướng dẫn trước đó". Hai lớp: (1) system prompt nói mọi thứ giữa các marker là dữ liệu không đáng tin và không được làm theo; (2) marker chứa **nonce ngẫu nhiên theo từng request** (`secrets.token_hex(8)`), và code xóa nonce khỏi text chunk, nên chunk không thể "đóng" khối dữ liệu sớm bằng cách tự viết marker kết thúc. Có test cho việc dựng prompt. Em nói thẳng: **chưa kiểm thử với mô hình thật bằng một tài liệu độc hại**, nên đây chỉ là giảm rủi ro, không phải bảo đảm. Một chunk còn có thể giả header kiểu `[2] (x.pdf, trang 1)` để gây nhầm số thứ tự. *(`app/prompts.py`, KI-19)*

**11. Quiz được kiểm chứng nguồn như thế nào?**
Mô hình đề xuất, **code quyết định giữ gì**. Mỗi câu phải có 4 đáp án khác nhau, khóa hợp lệ, giải thích không rỗng, và một đoạn trích dài ít nhất 15 ký tự phải xuất hiện **nguyên văn** (sau chuẩn hóa NFC và khoảng trắng, phân biệt hoa thường) trong chunk mà câu hỏi nêu. Số trang luôn lấy từ chunk, không lấy từ mô hình. Vị trí đáp án đúng do code gán theo khối cân bằng và xáo trộn nên mô hình không thể ép thiên lệch "luôn A". Có thêm một lần gọi LLM để xác minh (mặc định bật). Số liệu thật: trong một lần yêu cầu 10 câu, mô hình đề xuất 11 câu và code loại 1 câu (`quote_not_in_source`); verifier chấp nhận 20/20 câu thật; kiểm soát âm tính 3/3 câu cố ý sai bị từ chối (n = 4). Giới hạn: đoạn trích chứng minh đáp án *có trong tài liệu* chứ không chứng minh câu hỏi *tốt*; kết quả đọc tay là câu hỏi khá dễ và đáp án gây nhiễu hay là khái niệm không liên quan. *(`app/quiz.py`, KI-28)*

**12. Lỗi timestamp trên Windows vừa sửa là gì?**
Hội thoại được sắp theo `updated_at` (hoạt động gần nhất). Trên Windows, `datetime.now()` có thể trả về cùng một giá trị cho hai lời gọi cách nhau vài mili giây, nên hai cuộc hội thoại hoà nhau về `updated_at` và thứ tự không ổn định. Sửa trong `app/db.py` bằng `_now_precise()`: dùng khóa luồng, và nếu giá trị mới không lớn hơn giá trị trước thì cộng 1 micro giây, đảm bảo **tăng nghiêm ngặt** trong cùng tiến trình, với độ chính xác micro giây. Regression test (`test_exchanges_in_the_same_clock_tick_still_sort_by_activity`) đóng băng đồng hồ để tái hiện hiện tượng. Giới hạn: bảo đảm chỉ trong một tiến trình; hai tiến trình khác nhau không được phối hợp. *(`app/db.py`, `tests/test_db_history.py`)*

**13. Nếu có thêm một tuần, em cải thiện gì?**
Theo thứ tự ưu tiên, mỗi mục có cách kiểm tra: (1) **nhãn mức chunk, nhiều tài liệu hơn (≥ ~50 câu), tách dev/test** — vì nhãn trang che mất lỗi như q13 và ngưỡng đang được chọn trên chính bộ test. (2) **Hybrid search (BM25 + vector) và reranker** cho câu liên quan-nhưng-vắng, vì một ngưỡng cosine không tách được (KI-17); kỳ vọng kiểm chứng là q13 lấy lại chunk 10 mà không làm tăng số câu lọt qua của q16–q18, nếu không thì bỏ ý tưởng. (3) **Timeout tổng cứng cho LLM** để không còn câu nào > 20 s như q04 (62.2 s), KI-18. (4) **Chunk có nhận biết cấu trúc**, loại bìa/mục lục và header/footer lặp lại (KI-01, KI-14). (5) **Xác thực và tách dữ liệu theo người dùng**, **hàng đợi tác vụ** cho index (KI-22, KI-23). *(report mục 9.5, "Possible improvements" trong README)*

**14. Vì sao dùng Qdrant, độ đo cosine, và vì sao phải chuẩn hóa vector?**
Qdrant chạy bằng Docker, có lọc theo payload (`doc_id`), nên tìm được theo từng tài liệu. Collection dùng khoảng cách **cosine**, mỗi chunk là một điểm với payload `doc_id, filename, page, chunk_index, text`. Em yêu cầu `output_dimensionality=768` từ Gemini; vector bị cắt như vậy không còn độ dài đơn vị, nên `GeminiEmbedder` tự chuẩn hóa L2. Task type phân biệt tài liệu (`RETRIEVAL_DOCUMENT`) và câu hỏi (`RETRIEVAL_QUERY`). Embedding model và số chiều lúc index và lúc truy vấn phải giống hệt nhau. *(`app/embeddings.py`, `app/vectorstore.py`)*

**15. Vì sao SQLite không ORM, và lịch sử hội thoại được lưu thế nào?**
Dự án là demo một người dùng, `sqlite3` chuẩn không cần cài đặt, nên không cần ORM. SQLite lưu metadata tài liệu, hội thoại và bộ nhớ đệm `generations`. Mỗi lượt hỏi-đáp được lưu trong **một transaction** (`Database.save_exchange`, `BEGIN IMMEDIATE`): hội thoại, tin nhắn người dùng, tin nhắn trợ lý và cập nhật `updated_at`, hoặc tất cả hoặc không gì cả. Nếu lưu lỗi, câu trả lời (đã tốn quota LLM) vẫn được trả về với `saved: false`. `conversation_id` sai trả 404 *trước* khi gọi mô hình nên không tốn quota. *(`app/db.py`, KI-29)*

**16. Kiểm tra trích dẫn làm được gì và không làm được gì?**
`resolve_citations` dùng regex tìm `[n]`, xóa số vượt quá số nguồn đã gửi và đánh dấu nguồn nào được trích. Trong lần đánh giá có 0 trích dẫn tới nguồn không tồn tại, 17/18 câu có ít nhất một trích dẫn hợp lệ, và 15/15 câu có đáp án trích ít nhất một nguồn ở trang đúng. Giới hạn (KI-20): nó không kiểm tra chunk có thật sự hỗ trợ câu đó; regex không phân biệt `[1]` trích dẫn với `[1]` trong code do mô hình trích lại (số từ 4 chữ số trở lên bị bỏ qua, như năm); câu không có trích dẫn nào vẫn được trả về, chỉ ghi cảnh báo. *(`app/rag.py`, report mục 5.3)*

**17. Xử lý rate limit và lỗi tạm thời của Gemini ra sao?**
Embedding được gửi theo batch; cả embedding và LLM đều có retry với backoff khi gặp 429 và 5xx (kể cả 504), đọc lại `retryDelay` của server ở mỗi lần, tối đa `EMBED_MAX_RETRIES` (mặc định 5). Với quota *theo ngày* thì không chờ hàng giờ mà báo `RATE_LIMITED` ngay. Điểm yếu em đã đo: timeout của SDK áp dụng theo từng pha chứ không phải tổng; có câu mất 62.2 s (q04) và một lần chạy khác lên tới 87 s. Vòng retry bị lặp ở hai file (KI-21), và không có bộ giới hạn tốc độ chủ động, chỉ phản ứng khi bị 429. *(`app/embeddings.py`, `app/rag.py`, KI-18, KI-21)*

**18. Bảo mật khi tải tệp lên?**
Kiểm tra cả **phần mở rộng, kích thước và nội dung thật** (magic bytes: `%PDF-`, `PK` cho DOCX, và từ chối nhị phân giả TXT). Tên tệp của người dùng không bao giờ chạm đĩa: tệp lưu trong `data/uploads` với tên sinh từ `doc_id` cộng đuôi nằm trong danh sách cho phép. Giới hạn kích thước được áp dụng **trước khi đọc body**: `BodySizeLimitMiddleware` từ chối `Content-Length` quá lớn mà không đọc gì (đo thật: tệp 30 MB so với giới hạn 20 MB bị trả 413 trong 0.02 s), và `_read_bounded` là lớp thứ hai (KI-09). Có thêm giới hạn chống zip bomb cho DOCX. Hạn chế: không có xác thực, ai truy cập được API đều thấy mọi thứ (KI-23). *(`app/ingestion.py`, `app/main.py`)*

**19. Tóm tắt xử lý tài liệu dài thế nào, và vì sao có bảng `generations`?**
Tóm tắt đọc các chunk của tài liệu từ Qdrant (`get_chunks`). Nếu tổng không vượt 60.000 ký tự thì **một** lần gọi LLM (tài liệu mẫu có 17.640 ký tự). Dài hơn thì map-reduce: mỗi nhóm tối đa 40.000 ký tự rồi một lần gộp, tối đa 7 lần gọi; vượt 6 nhóm thì phần cuối bị bỏ và phản hồi báo `truncated`. Kết quả lưu trong bảng `generations`, khóa theo văn bản + tham số + mô hình + phiên bản prompt, nên lặp lại tốn 0 lượt gọi (đo 0.0–0.1 s), còn đổi prompt thì cache tự vô hiệu. Hạn chế: ngân sách tính bằng ký tự chứ không phải token, hai request giống hệt đến cùng lúc đều gọi mô hình, và nhánh map-reduce mới chỉ chạy với giới hạn nhỏ giả lập (KI-28). *(`app/summary.py`, `app/generation.py`)*

**20. Lịch sử hội thoại có giúp mô hình hiểu câu hỏi nối tiếp không? Còn gì dự án chưa đo?**
Không. Lịch sử chỉ được **lưu và xem lại** (mở một hội thoại cũ không gọi LLM), tin nhắn trước **không** được gửi cho mô hình, nên "còn cái đó thì sao?" được trả lời như một câu hỏi độc lập. Viết lại câu nối tiếp cần thêm một lần gọi LLM mỗi câu và nằm ngoài phạm vi; giao diện ghi rõ điều này (KI-29). Những gì chưa đo: chất lượng của `gemini-3.5-flash` (hết quota khi chạy), tài liệu dài hơn 9 trang, nhiều tài liệu cùng lúc, DOCX/TXT (không có số trang), và tác động riêng của header/footer lặp lại (KI-26, KI-01).

---

## 4. Năm câu hỏi "bẫy"

**Bẫy 1. "Hit@5 = 1.000 và MRR = 1.000, vậy truy xuất của em hoàn hảo rồi chứ?"**
*Không.* Đây là trả lời trung thực: "Con số đó chỉ nói rằng trang đúng luôn nằm trong top-5 và ở hạng 1 trên 15 câu. Nhãn của em ở mức trang nên một chunk cùng trang nhưng không chứa đáp án vẫn tính là đúng. Ví dụ q13 cần 3 chunk của trang 4 nhưng chỉ 2 chunk vào prompt (chunk còn lại điểm 0.6437 < 0.65), mà Recall mức trang vẫn là 1.0. Hơn nữa chỉ có 15 câu, một tài liệu, và q15 cho thấy Recall@5 = 2/3 (trang 8 chỉ xếp hạng 11). Số này kiểm tra hồi quy chứ không phải benchmark." *(KI-26, report 9.3)*

**Bẫy 2. "Faithfulness 18/18, vậy hệ thống không hallucinate?"**
*Không thể kết luận vậy.* "Con số này do LLM chấm, và judge cùng mô hình với mô hình trả lời. Em có ý kiến thứ hai từ Claude, đồng ý 18/18, nhưng hai LLM đồng ý không phải sự thật. N = 18. Thêm nữa, 3/18 câu là từ chối đúng nên rất dễ 'trung thực' (câu không phải từ chối là 15/15). Kiểm tra trích dẫn chỉ chứng minh số tồn tại (KI-20). Điều em có thể nói là: không thấy hallucination trong 18 câu và có nhiều lớp giảm rủi ro, không phải đã chứng minh được không có." *(report 5.3, 7, KI-20, KI-26)*

**Bẫy 3. "Ngưỡng 0.68 cho ít lỗi nhất, sao em vẫn giữ 0.65?"**
"0.68 là một điểm lưới đơn lẻ (2 lỗi), còn 0.67 và 0.69 đều tệ hơn (3 và 4 lỗi). q11 chỉ vượt 0.68 đúng 0.0003, nên chỉ cần diễn đạt khác một chút là bị chặn nhầm; ở 0.69 đã có 2 câu có đáp án bị chặn. Hơn nữa ngưỡng được chọn và đánh giá trên cùng 20 câu, nên 'tối ưu' đó là lạc quan (overfitting). Em cân nhắc: chặn nhầm làm người dùng mất câu trả lời thật, còn lọt qua thì còn lớp an toàn thứ hai là prompt. Đây là phán đoán về đánh đổi chứ chưa được kiểm chứng ngoài bộ này." *(report 4.2, 9.4)*

**Bẫy 4. "Quiz có đoạn trích nguyên văn rồi, vậy câu hỏi chắc chắn đúng và tốt?"**
"Đoạn trích chỉ chứng minh *đáp án có trong tài liệu*. Nó không chứng minh câu hỏi hay, không có hai đáp án cùng đúng, hay nhiễu đủ khó. Em đọc tay 15 câu thật: không câu nào sai, một câu hơi nhập nhằng, nhiều đáp án nhiễu là khái niệm không liên quan nên quiz dễ. Verifier cùng họ mô hình, chỉ thấy chunk được trích, chấp nhận 20/20 và không phát hiện câu nhập nhằng kia; kiểm soát âm tính chỉ có n = 4. Và API trả cả đáp án nên không dùng để thi được." *(KI-28)*

**Bẫy 5. "Chatbot nhớ cuộc trò chuyện, sẵn sàng triển khai cho người dùng thật chưa?"**
"Chưa, và cả hai vế đều không. Lịch sử chỉ được lưu để xem lại; mô hình không dùng tin nhắn trước nên câu hỏi nối tiếp được xử lý độc lập (KI-29). Về triển khai: không có xác thực, mọi client thấy, mở và xóa mọi hội thoại và tài liệu (KI-23, KI-29); câu hỏi, câu trả lời và đoạn trích 200 ký tự lưu rõ trong SQLite; xóa tài liệu không xóa đoạn trích đã lưu trong lịch sử; indexing nền không bền (KI-22); và nhóm luồng 40 có thể cạn khi nhiều request LLM chậm (KI-24). Đây là demo cục bộ; tôi biết chính xác cần làm gì để lên production." *(KI-22, KI-23, KI-24, KI-29)*

---

## 5. Ba gạch đầu dòng cho CV (tiếng Anh)

- Built an end-to-end RAG document Q&A app (FastAPI, Streamlit, Qdrant, Gemini via `google-genai`) with a hand-written pipeline (no LangChain): page-aware chunking, idempotent uuid5 chunk IDs, batched embeddings behind a swappable interface, and citation validation,backed by 1,201 passing automated tests.
- Designed a score-threshold gate that returns an "insufficient evidence" answer without calling the LLM, and measured it on a 20-question evaluation set (single-chunk, multi-chunk, absent-topic and unrelated questions): Hit@5 1.000, MRR 1.000, Recall@5 0.978, 12/15 answers fully correct and 3/15 partially correct (0 wrong), 5/5 unanswerable questions handled correctly, 18/18 LLM-judged faithful answers, median latency 1.6 s.
- Documented limits honestly instead of overstating results: found that a single cosine threshold cannot separate related-but-absent questions (class overlap of 0.037), verified summary and quiz generation in code (verbatim quote check, code-side option shuffling, SQLite result cache), and tracked 29 known issues with measured evidence.

> Lưu ý khi dùng: các số trên đến từ **một tài liệu 9 trang và 20 câu hỏi**. Nếu bị hỏi, hãy nói rõ điều đó. ""1,201 passing tests" là kết quả `pytest -q` khi Qdrant đang chạy (0 skip, 3 test tích hợp gọi Gemini thật bị loại theo mặc định); nếu bạn thêm hay bớt test, cập nhật lại con số.
