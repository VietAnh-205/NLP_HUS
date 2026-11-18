# LAB 09 — Giả thuyết và Dự đoán trước thực nghiệm (Prediction)


## 1. Prediction 1 — Pretrained Transformer vs. Simple Word-Feature Classifier

### Câu hỏi:
Nếu sử dụng Pretrained Transformer thay vì một classifier đơn giản dựa trên word features (TF-IDF / đặc trưng hình thái token + Logistic Regression), hiệu năng NER sẽ thay đổi như thế nào?

- **Prediction:**
  **Entity-level F1 sẽ tăng mạnh (dự kiến tăng từ khoảng $68\% - 75\%$ lên trên $89\% - 92\%$)**, đặc biệt cải thiện vượt bậc ở các thực thể phụ thuộc ngữ cảnh (context-dependent entities) và ranh giới thực thể đa token.
- **Reason:**
  1. Classifier tuyến tính dựa trên đặc trưng từ cục bộ (viết hoa, tiền tố/hậu tố, cửa sổ từ liền kề) rất dễ nhầm lẫn các từ có cùng hình thức bề mặt (*"Apple"* là ORG hay trái cây, *"Washington"* là PER hay LOC) và không mô hình hóa được sự phụ thuộc chuỗi giữa các thẻ `B-X` và `I-X`.
  2. Pretrained Transformer (BERT) sử dụng Self-Attention hai chiều, tạo ra **biểu diễn ngữ cảnh động (contextual representation)** cho từng token dựa trên toàn bộ câu, đồng thời đã tích lũy tri thức khổng lồ về tên người, tổ chức, địa danh từ giai đoạn tiền huấn luyện trên Wikipedia và BooksCorpus.
- **Confidence:** **98%**

---

## 2. Prediction 2 — Thực thể đa token dài vs. Thực thể đơn token

### Câu hỏi:
Các thực thể dài nhiều token (ví dụ: *United Nations Security Council*) sẽ dễ hay khó hơn thực thể chỉ có một token (ví dụ: *Microsoft*)? Giải thích.

- **Prediction:**
  **Thực thể dài nhiều token sẽ khó hơn đáng kể** và có Entity-level F1 giảm dần theo độ dài thực thể ($1 \text{ token} > 2 \text{ tokens} > 3 \text{ tokens} > 4+ \text{ tokens}$).
- **Reason:**
  1. **Tiêu chuẩn khớp chính xác (Exact Match Penalty):** Ở mức entity-level, một thực thể $k$ tokens chỉ được tính là $TP$ nếu mô hình dự đoán đúng đồng thời cả $k$ nhãn BIO (đúng điểm mở đầu `B-X`, toàn bộ chuỗi `I-X` và đúng điểm kết thúc). Chỉ cần sai 1 token biên (ví dụ bỏ sót `Council` hoặc tách `United Nations` riêng), toàn bộ thực thể bị tính vừa là $1\ FP$ vừa là $1\ FN$.
  2. **Tích lũy xác suất lỗi:** Nếu xác suất đúng mỗi token độc lập xấp xỉ $p < 1$, xác suất đúng toàn cụm xấp xỉ $p^k$ giảm theo hàm mũ khi $k$ tăng.
  3. **Từ chức năng bên trong thực thể dài:** Thực thể dài thường chứa giới từ, mạo từ, liên từ (*"of", "the", "and"* trong *"University of California Berkeley"*), dễ gây đứt gãy chuỗi `I-ORG` sang `O`.
- **Confidence:** **95%**

---

## 3. Prediction 3 — Thực thể hiếm (Rare / Unseen Entities)

### Câu hỏi:
Các thực thể hiếm xuất hiện trong training data có khả năng gây lỗi nhiều hơn không?

- **Prediction:**
  **Có, tỷ lệ lỗi (đặc biệt là False Negative và sai loại thực thể) sẽ cao hơn rõ rệt** đối với các thực thể hiếm hoặc chưa từng xuất hiện trong tập train (OOV / long-tail entities).
- **Reason:**
  1. **Thiếu mẫu ghi nhớ từ vựng (Lexical Prior):** Với thực thể phổ biến (*"London", "Reuters", "Bill Clinton"*), mô hình vừa có tín hiệu ngữ cảnh vừa có xác suất tiên nghiệm từ vựng mạnh. Với thực thể hiếm (tên cầu thủ địa phương, công ty nhỏ, tên viết tắt ít gặp), mô hình phải dựa hoàn toàn vào ngữ cảnh cú pháp xung quanh.
  2. **Hiệu ứng phân mảnh Subword (Subword Fragmentation):** Các tên riêng hiếm thường không nằm trọn trong từ điển WordPiece $30,522$ từ nên bị chia vụn thành 3–5 subwords (`##...`), làm tăng độ khó trong quá trình căn chỉnh nhãn (label alignment) và tổng hợp ngữ nghĩa toàn từ.
  3. **Nhầm lẫn giữa `ORG` và `MISC`/`PER`:** Khi không biết trước bản chất thực thể, mô hình dễ phân loại sai giữa tên người và tên công ty đặt theo tên người (ví dụ: *"Goldman Sachs"*, *"Ford"*).
- **Confidence:** **95%**
