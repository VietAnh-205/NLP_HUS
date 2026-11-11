

## 1. Tổng quan Lab

Trong các bài thực hành trước, chúng ta đã tiếp cận quá trình huấn luyện mô hình học máy từ đầu (Training from Scratch):
- **LAB 01 - 04:** Mô hình không gian vector túi từ (TF-IDF) và véc-tơ trung bình (Word2Vec) với bộ phân loại tuyến tính.
- **LAB 05:** Mạng hồi quy tuần tự (Simple RNN, LSTM, GRU).
- **LAB 06 - 07:** Cơ chế Self-Attention và khối Transformer Encoder tự xây dựng từ đầu.

Mặc dù kiến trúc Transformer (LAB 07) đạt hiệu năng ấn tượng ($F_1 \approx 87\%$), việc huấn luyện từ đầu trên một tập dữ liệu nhỏ ($2,000$ bài review) bộc lộ giới hạn nghiêm trọng: mô hình rất dễ rơi vào bẫy quá khớp (overfitting), đòi hỏi chi phí gán nhãn thủ công lớn và hoàn toàn không tận dụng được kho tri thức khổng lồ từ lượng văn bản không nhãn trên Internet.

Trong **LAB 08**, chúng ta chính thức bước sang kỷ nguyên mang tính cách mạng của NLP hiện đại: **Cơ chế Tiền huấn luyện & Tinh chỉnh (Pretraining & Fine-Tuning Paradigm)** với đại diện tiêu biểu là **BERT (Bidirectional Encoder Representations from Transformers)**:
1. **Tiền huấn luyện tự giám sát (Self-Supervised Pretraining):** Mô hình được huấn luyện trên hàng tỷ từ vựng với tác vụ Masked Language Modeling (MLM), học được biểu diễn ngữ nghĩa và cấu trúc cú pháp hai chiều sâu sắc.
2. **Subword Tokenization (WordPiece):** Giải quyết triệt để vấn đề từ chưa thấy (Out-Of-Vocabulary - OOV) bằng cách tách từ thành các phân đoạn từ con có tiền tố `##`.
3. **Tinh chỉnh hạ nguồn (Downstream Fine-Tuning):** Khởi tạo từ trọng số đã được tiền huấn luyện, chỉ cần cập nhật nhẹ nhàng trên tập dữ liệu mục tiêu để đạt hiệu năng vượt trội với lượng dữ liệu gán nhãn rất nhỏ.

---

## 2. Mục tiêu học tập cốt lõi (Learning Outcomes)

1. **LO1 — Pretraining vs. Fine-tuning:** Nắm vững sự khác biệt bản chất giữa huấn luyện từ đầu (Train from Scratch) và tinh chỉnh từ mô hình tiền huấn luyện (Pretraining & Fine-tuning).
2. **LO2 — Subword Segmentation:** Hiểu và phân tích thuật toán tách từ con WordPiece, phân biệt từ tự nhiên (words), token và phân đoạn subwords có tiền tố `##`.
3. **LO3 — Special Tokens & Masking:** Nắm vững vai trò cấu trúc của các token đặc biệt (`[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`, `[UNK]`) và chức năng của Attention Mask trong việc loại bỏ token đệm.
4. **LO4 — Masked Language Modeling (MLM):** Tính toán giải tích hàm mất mát Cross-Entropy Loss của bài toán MLM và phân tích độ nhạy của gradient khi xác suất dự đoán thay đổi.
5. **LO5 — Feature Extraction vs. Full Fine-tuning:** So sánh định lượng và lý giải bản chất giữa chế độ đóng băng Encoder (Frozen / Feature Extraction) và tinh chỉnh toàn phần (Full Fine-tuning).
6. **LO6 — Sample Efficiency:** Đánh giá bằng thực nghiệm đường cong hiệu quả dữ liệu (Data Efficiency Curve) khi nguồn dữ liệu gán nhãn bị thu hẹp ($100\% \to 10\%$).
7. **LO7 — Sequence Length Trade-off:** Đo lường chi phí tính toán $\mathcal{O}(T^2)$ và sự bão hòa hiệu năng khi tăng độ dài chuỗi cắt cụt $T \in \{64, 128, 256\}$.
8. **LO8 — Domain Shift & Generalization:** Kiểm nghiệm giới hạn của Pretrained Model khi chuyển dịch từ miền dữ liệu huấn luyện (In-domain) sang miền dữ liệu kiểm thử khác biệt (Out-of-domain).
9. **LO9 — Confidence vs. Correctness:** Nhận diện hiện tượng quá tự tin (Overconfidence Miscalibration) và chứng minh bằng số liệu rằng độ tự tin cao không đồng nghĩa với dự đoán đúng.
10. **LO10 — Qualitative Error Diagnosis:** Phân tích định tính các ca lỗi điển hình do cấu trúc câu châm biếm (sarcasm) hoặc phủ định kép phức tạp, phân biệt rõ giữa tương quan số học và lập luận nhân quả.

---

## 3. Cấu trúc thư mục nộp bài (Deliverables)

Thư mục `w8/` bao gồm đầy đủ **12 tệp tin** theo đúng quy định tại **Mục 25. Deliverables**:

```
w8/
├── README.md                           # Báo cáo tổng quan, kết quả thực nghiệm và hướng dẫn sử dụng
├── calculations.md                     # Lời giải ngắn gọn, chuẩn xác 2 bài tập tính toán (MLM Loss & Tokenization analysis)
├── prediction.md                       # Thiết lập 3 giả thuyết nghiên cứu định lượng kèm lý giải cơ chế và mức độ tin cậy
├── tokenization.ipynb                  # Notebook thực nghiệm Subword Tokenization, Special Tokens, Padding, Attention Mask & MLM Loss
├── pretrained_classification.py        # Module PyTorch cài đặt BertForSentimentClassification (hỗ trợ cả Frozen và Fine-tuning)
├── fine_tuning.ipynb                   # Notebook thực nghiệm hoàn chỉnh từ Exp 1 đến Exp 5, Application, Confidence & Reproducibility
├── results.csv                         # Bảng số liệu thực nghiệm định lượng chi tiết của tất cả các thí nghiệm
├── data_efficiency.png                 # Đồ thị đường cong hiệu quả dữ liệu (10% -> 100%) so sánh Pretrained BERT vs From Scratch
├── confusion_matrix.png                # Ma trận nhầm lẫn của Pretrained BERT fine-tuned trên tập kiểm thử IMDb
├── confidence_analysis.png             # Biểu đồ phân phối độ tự tin và sơ đồ kiểm chứng Confidence != Correctness
├── error_analysis.md                   # Bảng phân tích định tính chuyên sâu 8 trường hợp dự đoán sai và kiểm chứng giả thuyết
└── reflection.md                       # Phân tích chuyên sâu Pretraining/Fine-tuning, trả lời câu hỏi Exp 1-5, đối chuẩn giả thuyết và AI Policy
```

---
