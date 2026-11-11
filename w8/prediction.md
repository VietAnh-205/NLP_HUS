

## 1. Prediction 1 — Pretrained Transformer vs. Train from Scratch trên IMDb

### Câu hỏi:
Một pretrained Transformer (BERT) được fine-tune trên IMDb có khả năng đạt kết quả tốt hơn một Transformer nhỏ được train from scratch trên cùng dữ liệu không?

- **Prediction:**
  **Có, vượt trội hoàn toàn.** Pretrained Transformer sau khi fine-tune sẽ đạt độ chính xác ($F_1$-score) cao hơn đáng kể (dự kiến $89\% - 93\%$) so với Transformer train from scratch (chỉ đạt $\sim 85\% - 87\%$).
- **Reason:**
  1. Mô hình train from scratch chỉ tiếp cận 2,000 bài review IMDb, phải học đồng thời cả ngữ pháp cơ bản của tiếng Anh, cấu trúc biểu diễn từ, lẫn ranh giới phân loại cảm xúc. Số lượng mẫu quá nhỏ dễ dẫn đến hiện tượng quá khớp (overfitting) các đặc trưng ngẫu nhiên.
  2. Pretrained BERT đã được huấn luyện tự giám sát (Self-Supervised Learning với MLM và NSP) trên kho ngữ liệu khổng lồ (BooksCorpus $800$ triệu từ + English Wikipedia $2,500$ triệu từ). Mô hình đã tích lũy sẵn biểu diễn ngữ nghĩa, cấu trúc cú pháp hai chiều sâu sắc; quá trình fine-tuning chỉ cần điều chỉnh nhẹ trọng số để thích ứng với bài toán phân loại cảm xúc downstream.
- **Expected Evidence:**
  - Accuracy và $F_1$ của Pretrained BERT cao hơn Transformer from scratch từ $3\% - 6\%$.
  - Đường cong Validation Loss của Pretrained BERT hội tụ nhanh hơn (ngay từ Epoch 2 - 3) và ổn định hơn.

---

## 2. Prediction 2 — Feature Extraction (Frozen Encoder) vs. Fine-tuning toàn bộ

### Câu hỏi:
Nếu freeze toàn bộ pretrained encoder và chỉ huấn luyện classification head:
$$\text{BERT Encoder (Frozen)} \longrightarrow \text{Classification Head (Trainable)} \longrightarrow \text{Prediction}$$
kết quả sẽ:
- **A.** Tốt hơn fine-tuning toàn bộ
- **B.** Tương đương
- **C.** Kém hơn
- **D.** Không thể dự đoán

- **Prediction:** **Lựa chọn C — Kém hơn.**
- **Reason:**
  1. Trong chế độ **Frozen (Feature Extraction)**, các trọng số của BERT được giữ nguyên cố định như một bộ trích xuất đặc trưng tĩnh. Vector đại diện `[CLS]` ở tầng cuối cùng chứa đựng tri thức ngôn ngữ tổng quát của quá trình MLM, chưa được định hướng tối ưu hóa riêng cho miền từ vựng phân loại phim ảnh (IMDb domain).
  2. Trong chế độ **Fine-tuning toàn bộ**, các tầng Attention và FFN của Transformer đều nhận tín hiệu gradient chảy ngược từ hàm mất mát phân loại. Mô hình có thể tinh chỉnh các trọng số chú ý để tập trung cao độ vào các từ ngữ bộc lộ cảm xúc then chốt (*"masterpiece"*, *"terrible"*, *"boring"*), giúp không gian biểu diễn phân tách hai lớp rõ rệt hơn.
- **Expected Evidence:**
  - Chế độ Fine-tuning sẽ đạt $F_1 \approx 90\% - 92\%$, cao hơn chế độ Frozen Encoder ($F_1 \approx 85\% - 87\%$).
  - Thời gian huấn luyện mỗi epoch của chế độ Frozen sẽ nhanh hơn đáng kể vì không cần tính gradient qua 12 tầng Transformer.

---

## 3. Prediction 3 — Hiệu quả sử dụng dữ liệu trong điều kiện thiếu dữ liệu (Low-Resource Data Efficiency)

### Câu hỏi:
Nếu giảm tỷ lệ dữ liệu huấn luyện ($100\% \to 50\% \to 25\% \to 10\%$):
Mô hình Pretrained hay mô hình Train from Scratch sẽ bị sụt giảm hiệu năng mạnh hơn?

- **Prediction:**
  **Mô hình Train from Scratch sẽ bị sụt giảm hiệu năng nghiêm trọng hơn rất nhiều** so với Pretrained Model.
- **Reason:**
  1. **Vai trò khởi tạo tri thức nền tảng (Inductive Bias & Semantic Priors):**
     - Khi chỉ có $10\%$ dữ liệu (chỉ khoảng 140 bài review để huấn luyện), mô hình Train from Scratch không có đủ dữ liệu để học được cấu trúc phân phối của từ vựng tiếng Anh. Số tham số hàng trăm nghìn hoặc hàng triệu trọng số sẽ nhanh chóng "học vẹt" 140 mẫu này, dẫn tới Overfitting cực đoan và hiệu năng trên tập test sụp đổ về mức ngẫu nhiên ($F_1 \approx 60\% - 68\%$).
     - Ngược lại, Pretrained BERT đã có sẵn không gian vector từ và hiểu biết ngữ nghĩa hoàn chỉnh. Với 140 mẫu, mô hình chỉ cần tinh chỉnh lại lớp phân loại tuyến tính cuối cùng để phân định cực tính, do đó vẫn duy trì được hiệu năng ấn tượng ($F_1 \approx 83\% - 86\%$).
- **Expected Evidence:**
  - Đồ thị đường cong hiệu quả dữ liệu (Data Efficiency Curve) của Pretrained Model sẽ có độ dốc phẳng hơn rất nhiều so với độ dốc rơi thẳng đứng của Model Train from Scratch khi dữ liệu giảm về $10\%$.

