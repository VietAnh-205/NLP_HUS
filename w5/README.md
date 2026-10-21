# LAB 05 — Sequence Modeling: Simple RNN, LSTM & GRU

---

## 1. Tổng quan Lab

Trong các bài thực hành trước, chúng ta đã tiếp cận các phương pháp biểu diễn văn bản dưới dạng véc-tơ cố định (Fixed-size Document Vector):
- **LAB 01:** Mô hình không gian véc-tơ túi từ thưa (Sparse Bag-of-Words & TF-IDF).
- **LAB 02:** Mô hình ngôn ngữ N-gram dựa trên xác suất chuỗi cục bộ Markov.
- **LAB 03:** Biểu diễn từ ngữ nghĩa dày đặc không phụ thuộc ngữ cảnh (Static Word2Vec Embeddings).
- **LAB 04:** Phân loại văn bản với mô hình túi từ véc-tơ trung bình ($\vec{x} = \frac{1}{T} \sum_{t=1}^T \vec{w}_t$).

Mặc dù các phương pháp trên đạt hiệu năng phân loại khả quan trên các văn bản dài có từ khóa rõ ràng, chúng hoàn toàn **bỏ qua trật tự từ (Word Order)** và **ngữ cảnh động (Dynamic Context)**. Một câu như *"dog bites man"* và *"man bites dog"* sẽ tạo ra véc-tơ trung bình hoàn toàn đồng nhất; hay cấu trúc phủ định phức tạp (*"not good"* vs *"not bad"*) rất dễ bị triệt tiêu ngữ nghĩa khi tính trung bình cộng toàn văn bản.

Trong **LAB 05**, chúng ta chính thức bước sang kỷ nguyên **Sequence Modeling**:
$$\vec{w}_1 \to \vec{w}_2 \to \dots \vec{w}_T \implies \vec{h}_1 \to \vec{h}_2 \to \dots \to \vec{h}_T$$
với cơ chế cập nhật trạng thái ẩn:
$$\vec{h}_t = f(\vec{x}_t, \vec{h}_{t-1})$$

Quy trình nghiên cứu và thực nghiệm trong Lab 05 gồm:
> Tính toán giải tích một bước RNN & LSTM Gates → Phân tích lan truyền thông tin & Vanishing Gradient → Thiết lập 5 giả thuyết dự đoán định lượng → Khảo sát phân phối độ dài văn bản & Chiến lược Padding/Truncation → Tự cài đặt Simple RNN Cell từ đầu bằng NumPy → Huấn luyện so sánh 3 kiến trúc Recurrent (Simple RNN, LSTM, GRU) trên IMDb Movie Reviews → Khảo sát thực nghiệm Long-Range Dependency & Vị trí từ khóa → Đánh giá Confusion Matrix, Learning Curves & Overfitting → Phân tích định tính 7 trường hợp đúng/sai → Tổng kết so sánh toàn diện và đề xuất bước chuyển sang Attention/Transformer.

---

## 2. Mục tiêu học tập cốt lõi (Learning Outcomes)

1. **LO1 — Sequence Modeling:** Nắm vững sự khác biệt bản chất giữa Document Representation tĩnh (Fixed-size Bag-of-Words/Word2Vec) và Sequence Representation động (Recurrent State Accumulation).
2. **LO2 — RNN Computation Graph:** Hiểu và tính toán chính xác bằng tay quá trình cập nhật trạng thái ẩn $\vec{h}_t = \tanh(W_{xh}\vec{x}_t + W_{hh}\vec{h}_{t-1} + \vec{b}_h)$ và phân loại $\hat{y} = \text{softmax}(W_{hy}\vec{h}_T + \vec{b}_y)$.
3. **LO3 — Word Order Sensitivity:** Giải thích bằng cơ chế toán học tại sao RNN có khả năng phân biệt trật tự chuỗi (*"dog bites man"* vs *"man bites dog"*) và ghi nhận đảo chiều trạng thái qua cấu trúc phủ định (*"not like"*).
4. **LO4 — Vanishing & Exploding Gradients:** Lý giải trực quan và toán học về hiện tượng tiêu biến/bùng nổ gradient qua chuỗi tích Jacobian $\prod_{k=t+1}^T \frac{\partial \vec{h}_k}{\partial \vec{h}_{k-1}}$ trong thuật toán Backpropagation Through Time (BPTT).
5. **LO5 — LSTM Gating Mechanism:** Nắm vững cấu trúc 3 cổng (Forget Gate $f_t$, Input Gate $i_t$, Output Gate $o_t$) cùng Cell State $c_t$ độc lập dạng tuyến tính cộng dồn để bảo toàn gradient tầm xa.
6. **LO6 — GRU Simplification:** So sánh cấu trúc gọn nhẹ của GRU (kết hợp Forget và Input thành Update Gate $z_t$, loại bỏ Cell State độc lập) so với LSTM.
7. **LO7 — Sequence Preprocessing:** Phân biệt bản chất của Padding (kèm Padding Mask) và Truncation, hiểu rõ trade-off giữa thông tin lưu giữ và chi phí tính toán $\mathcal{O}(T \cdot d^2)$.
8. **LO8 — Empirical Performance Comparison:** So sánh khách quan hiệu năng (Accuracy, Precision, Recall, $F_1$, thời gian huấn luyện, số lượng tham số) giữa Baseline BoW, Simple RNN, LSTM và GRU.
9. **LO9 — Model Diagnosis:** Phân tích biểu đồ học (Learning Curves) để nhận diện sớm hiện tượng Overfitting khi train loss giảm sâu nhưng validation loss đảo chiều tăng.
10. **LO10 — Error Analysis & The Attention Transition:** Phân tích định tính các ca lỗi do phụ thuộc tầm xa, câu châm biếm (sarcasm) hoặc mất mát thông tin khi cắt cụt (truncation), chỉ ra điểm nghẽn cổ chai thông tin (Information Bottleneck) của recurrent state và lý do ra đời của Attention Mechanism.

---

## 3. Cấu trúc thư mục nộp bài (Deliverables)

Thư mục `w5/` bao gồm đầy đủ **10 tệp tin** theo đúng yêu cầu tại **Mục 29. Deliverables**:

```
w5/
├── README.md                      # Báo cáo tổng quan, kết quả thực nghiệm và hướng dẫn sử dụng
├── calculations.md                # Lời giải chi tiết 5 bài toán giải tích (RNN step, Information flow, Order, BPTT, LSTM gates)
├── prediction.md                  # Thiết lập 5 giả thuyết nghiên cứu có mức độ tin cậy và lý giải cơ chế
├── rnn_cell.py                    # Module thuần NumPy cài đặt SimpleRNNCell, Sequence Forward và Classifier
├── sequence_classification.ipynb  # Notebook thực nghiệm hoàn chỉnh từ Mục 11 đến Mục 26 (code sạch, không comment)
├── results.csv                    # Bảng số liệu thực nghiệm định lượng chi tiết của các mô hình
├── learning_curves.png            # Đồ thị đường cong học tập (Loss và Validation Accuracy qua 10 epochs)
├── confusion_matrix.png           # Ma trận nhầm lẫn của mô hình tối ưu (LSTM) trên tập kiểm thử
├── error_analysis.md              # Bảng phân tích định tính chuyên sâu 7 trường hợp đúng/sai
└── reflection.md                  # Bảng so sánh 5 mô hình, trả lời 8 câu hỏi vấn đáp cá nhân và AI Policy
```

---
