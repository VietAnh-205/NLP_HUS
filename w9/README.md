# LAB 09 — Named Entity Recognition (NER), Token Classification & Information Extraction

---

## 1. Tổng quan Lab

Nếu **LAB 08** tập trung vào bài toán phân loại cấp chuỗi (**Sequence Classification**: $X \to y$, gán 1 nhãn cho toàn bộ câu thông qua vector `[CLS]`), thì **LAB 09** chuyển sang bài toán gán nhãn cấp độ từng từ (**Token Classification**: $X = (x_1, \dots, x_T) \to Y = (y_1, \dots, y_T)$) với ứng dụng cốt lõi là **Nhận diện Thực thể có Tên (Named Entity Recognition — NER)** trên chuẩn **CoNLL-2003** (`PER`, `ORG`, `LOC`, `MISC`, `O`).

Nội dung xuyên suốt của **LAB 09** giải quyết 4 trụ cột kỹ thuật quan trọng nhất của NER hiện đại:
1. **Biểu diễn ranh giới thực thể bằng sơ đồ BIO:** Phân biệt rõ `B-X` (Beginning), `I-X` (Inside) và `O` (Outside) để tách bạch chính xác các thực thể đứng liền kề và các thực thể đa token dài.
2. **Căn chỉnh nhãn Subword (Label Alignment):** Xử lý sự lệch pha giữa nhãn word-level của dataset và subword tokens của WordPiece bằng `word_ids()` kết hợp chiến lược `IGNORE = -100` trong `CrossEntropyLoss`.
3. **Đánh giá chuẩn mực ở mức Thực thể (Entity-level Metrics):** Chứng minh vì sao Token Accuracy gây ngộ nhận nghiêm trọng và thực thi đánh giá Exact Span Match (Precision, Recall, F1) theo loại thực thể và độ dài thực thể.
4. **Ứng dụng Trích xuất Thông tin (Information Extraction — IE):** Chuyển đổi văn bản phi cấu trúc thành dữ liệu có cấu trúc JSON và kiểm định định lượng trên tập chuẩn 30 câu tin tức.

---

## 2. Mục tiêu học tập cốt lõi (12 Learning Outcomes)

1. **LO1:** Phân biệt rõ ràng Sequence Classification và Token Classification.
2. **LO2:** Giải thích bản chất và vai trò của bài toán Named Entity Recognition (NER).
3. **LO3:** Nắm vững và áp dụng chính xác hệ thống gán nhãn BIO (`B-X`, `I-X`, `O`).
4. **LO4:** Chuyển đổi chính xác chú thích thực thể (entity spans) thành chuỗi nhãn token-level.
5. **LO5:** Hiểu và cài đặt bài toán căn chỉnh (alignment) giữa word, subword token và label.
6. **LO6:** Thiết kế và triển khai Pretrained Transformer cho bài toán Token Classification ($Z = HW + b \in \mathbb{R}^{B \times T \times C}$).
7. **LO7:** Fine-tune và đánh giá đối chuẩn mô hình trên chuẩn CoNLL-2003.
8. **LO8:** Tính toán và diễn giải chính xác Precision, Recall, $F_1$ cho NER.
9. **LO9:** Phân biệt bản chất giữa Token-level metrics và Entity-level Exact Match metrics.
10. **LO10:** Thực hiện phân tích lỗi định tính chuyên sâu (5 FP, 5 FN, 5 Boundary Errors).
11. **LO11:** Nhận diện đầy đủ các nhóm lỗi do tokenization, context, entity boundary, rare entity và domain shift.
12. **LO12:** Xây dựng và đánh giá pipeline Information Extraction biến văn bản thô thành JSON có cấu trúc.

---

## 3. Cấu trúc thư mục nộp bài (Deliverables)

Thư mục `w9/` bao gồm đầy đủ **13 tệp tin** theo đúng quy định tại **Mục 29. Deliverables**:

```
w9/
├── README.md                    # Báo cáo tổng quan, bảng kết quả thực nghiệm và hướng dẫn sử dụng
├── calculations.md              # Lời giải ngắn gọn, đi thẳng trọng tâm 3 bài tập tính toán (BIO, Boundary, Tensor Shapes)
├── prediction.md                # Thiết lập 3 giả thuyết khoa học định lượng trước khi thực nghiệm
├── baseline_ner.py              # Baseline NER (Token features + Logistic Regression) và bộ đánh giá Entity-level chính xác (không comment)
├── tokenizer_alignment.ipynb    # Notebook thực nghiệm Subword Alignment (Strategy A/B, -100 Loss) và Experiment 1 trên 20 câu
├── ner_finetuning.ipynb         # Notebook thực nghiệm Token Classification, Experiments 2-5, Context Analysis và Mini IE
├── results.csv                  # Bảng tổng hợp kết quả định lượng (Baseline vs Pretrained BERT, Mini IE 30 câu)
├── entity_metrics.csv           # Bảng phân rã chi tiết theo Entity Type (PER/ORG/LOC/MISC) và Entity Length (1/2/3/4+)
├── entity_length_analysis.png   # Biểu đồ Entity-level F1 theo độ dài thực thể và theo từng loại thực thể
├── confusion_matrix.png         # Ma trận nhầm lẫn chuẩn hóa theo nhóm nhãn thực thể (O, PER, ORG, LOC, MISC)
├── error_analysis.md            # Phân tích định tính 15 ca lỗi thực tế (5 FP, 5 FN, 5 Boundary Errors)
├── information_extraction.py    # Pipeline trích xuất thông tin JSON và đánh giá định lượng trên 30 câu tin tức (không comment)
└── reflection.md                # Tổng kết so sánh LAB 08 vs LAB 09, kiểm chứng 3 giả thuyết và AI Usage Statement
```

---
