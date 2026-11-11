

## 1. Phân tích So sánh: Training from Scratch vs. Pretraining & Fine-Tuning

| Tiêu chí | Training from Scratch (LAB 07) | Pretraining & Fine-Tuning (LAB 08) |
| :--- | :--- | :--- |
| **Điểm khởi tạo trọng số** | Khởi tạo ngẫu nhiên hoàn toàn ($\theta \sim \mathcal{N}(0, \sigma^2)$) | Điểm checkpoint tiền huấn luyện tự giám sát ($\theta_{\text{pretrained}}$) |
| **Bản chất biểu diễn (Representation)** | Biểu diễn nông, chỉ tối ưu hóa cục bộ trên tập dữ liệu tác vụ nhỏ ($1,400$ mẫu) | Biểu diễn ngữ nghĩa phân tán sâu sắc, học từ kho ngữ liệu khổng lồ (3.3 tỷ từ) |
| **Lượng dữ liệu gán nhãn cần thiết** | Rất lớn (hàng chục nghìn đến hàng triệu mẫu để tránh Overfitting) | Rất nhỏ ($100 - 1,000$ mẫu vẫn đạt hiệu năng cao nhờ Sample Efficiency) |
| **Mục tiêu tối ưu hóa ban đầu** | Task Loss trực tiếp (Classification Cross-Entropy) | Masked Language Modeling (MLM) + Next Sentence Prediction (NSP) |
| **Khả năng khái quát hóa (Generalization)** | Kém trên các câu có cấu trúc lạ hoặc từ vựng hiếm | Xuất sắc, có khả năng xử lý subwords và ngữ cảnh hai chiều phong phú |
| **Chi phí tính toán tác vụ hạ nguồn** | Nhẹ về số tham số ($0.77M$), nhưng phải train nhiều epochs | Nặng về số tham số ($110M$), nhưng chỉ cần $2 - 3$ epochs là hội tụ |

---

## 2. Lời giải Đáp án cho các Câu hỏi Thực nghiệm Cốt lõi (Experiments 1 – 5)

### 2.1. Experiment 1 — Pretrained vs. Train from Scratch
1. **Representation khác nhau như thế nào?**
   - *From Scratch:* Không gian vector từ được học ngẫu nhiên từ đầu, chỉ mã hóa được tần suất đồng xuất hiện thô sơ trong 1,400 bài review phim, thiếu khả năng phân biệt từ đồng nghĩa / trái nghĩa tinh vi.
   - *Pretrained BERT:* Các tầng Transformer đã được tôi luyện để nhận diện các cấu trúc cú pháp hai chiều, phân biệt ngữ cảnh đa nghĩa (*"bank"* trong sông vs ngân hàng) và chia sẻ gốc từ qua WordPiece.
2. **Training Data cần thiết khác nhau như thế nào?**
   - Mô hình Train from Scratch cần kho dữ liệu có nhãn lớn gấp $50 - 100$ lần để đạt được ngưỡng $F_1 > 90\%$. Pretrained BERT đạt $F_1 = 92.28\%$ chỉ với 1,400 mẫu huấn luyện.
3. **Chi phí tính toán khác nhau như thế nào?**
   - Chi phí huấn luyện Pretrained BERT lớn hơn (42.1s vs 5.4s) do số lượng tham số lớn gấp 140 lần ($109.5M$ vs $0.77M$). Tuy nhiên, chi phí này chỉ là chi phí tinh chỉnh cục bộ; chi phí tiền huấn luyện đắt đỏ đã được chia đều cho cộng đồng nghiên cứu (Amortized Pretraining Cost).
4. **Kết quả có phù hợp với Prediction 1 không?**
   - Hoàn toàn phù hợp ($F_1$ tăng từ $86.80\% \to 92.28\%$).

### 2.2. Experiment 2 — Feature Extraction (Frozen) vs. Full Fine-Tuning
- **Vì sao Frozen Encoder vẫn thực hiện phân loại tốt ($F_1 = 86.55\%$)?**
  Bởi vì quá trình tiền huấn luyện tự giám sát (MLM) đã gom góp đại đa số thông tin ngữ nghĩa toàn câu vào vector ẩn `[CLS]`. Không gian ẩn này đã có tính chất phân tách tuyến tính tương đối tốt giữa các khái niệm ngữ nghĩa đối lập.
- **Vì sao Fine-Tuning toàn bộ cải thiện vượt trội ($+5.73\%$)?**
  Vì việc cho phép gradient chảy qua cả 12 tầng Transformer giúp các đầu Self-Attention tái cấu trúc lại không gian chú ý, tập trung sâu hơn vào các từ vựng bộc lộ cảm xúc đặc thù của miền phim ảnh, biến một biểu diễn ngôn ngữ tổng quát thành biểu diễn chuyên biệt cho tác vụ phân loại.

### 2.3. Experiment 3 — Hiệu quả Dữ liệu (Data Efficiency)
- **Pretraining có giúp mô hình sử dụng dữ liệu downstream hiệu quả hơn không?**
  - **Chắc chắn có.** Khi lượng dữ liệu huấn luyện giảm $10$ lần (từ $100\% \to 10\%$, chỉ còn 140 mẫu):
    - Pretrained BERT chỉ giảm nhẹ $6.08\%$ $F_1$ (vẫn đạt $86.20\%$).
    - Transformer from Scratch sụp đổ tới $24.30\%$ $F_1$ (rơi xuống $62.50\%$).
  - Đường cong Data Efficiency chứng minh rằng tri thức tiền huấn luyện đóng vai trò như một bộ khởi tạo tối ưu (Inductive Bias), giúp mô hình không bị quá khớp khi tài nguyên dữ liệu gán nhãn cực kỳ hạn chế.

### 2.4. Experiment 4 — Maximum Sequence Length Ablation
- **Tăng sequence length có luôn làm mô hình tốt hơn không?**
  - **Không luôn luôn.** Tăng $T$ từ $64 \to 128$ giúp $F_1$ tăng mạnh $+2.78\%$ vì nắm bắt được các luận điểm chốt hạ ở nửa sau bài phê bình.
  - Tuy nhiên, khi tăng tiếp từ $128 \to 256$, thời gian huấn luyện và dung lượng bộ nhớ tăng gấp đôi do độ phức tạp $\mathcal{O}(T^2)$ của ma trận Attention ($256^2 = 65,536$ phần tử), trong khi $F_1$ chỉ tăng không đáng kể $+0.67\%$. Chiều dài $T=128$ là điểm cân bằng hoàn hảo.

### 2.5. Experiment 5 — Trôi dạt Miền Dữ liệu (Domain Shift)
- **Pretrained Model có đồng nghĩa với việc mô hình đã "hiểu mọi domain" hay không?**
  - **Không.** Khi đưa mô hình fine-tuned trên IMDb sang kiểm thử trên tập đánh giá sản phẩm công nghệ (Tech Products), hiệu năng giảm ngay $9.13\%$ $F_1$ (từ $92.28\% \to 83.15\%$).
  - Pretrained model chỉ cung cấp nền tảng cú pháp và từ vựng chung. Khi phong cách hành văn và tiêu chí đánh giá thay đổi giữa các miền dữ liệu, mô hình vẫn cần được thích nghi hóa miền (Domain Adaptation) để duy trì độ chính xác.

---

## 3. Bảng đối chuẩn và Kiểm chứng Giả thuyết (Prediction Verification)

| STT | Giả thuyết dự đoán ban đầu | Dự đoán | Kết quả thực tế | Đánh giá | Cơ chế giải thích |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | Pretrained BERT vs. Scratch trên IMDb? | Pretrained BERT vượt trội hoàn toàn | $F_1 = 92.28\%$ vs. $86.80\%$ (Tăng $+5.48\%$) | **ĐÚNG** | Tri thức tiền huấn luyện sâu sắc trên 3.3 tỷ từ cung cấp không gian vector ưu việt hơn hẳn việc học từ 1,400 mẫu. |
| **2** | Frozen Encoder vs. Full Fine-Tuning? | Lựa chọn C (Kém hơn) | Frozen: $86.55\%$ vs. Fine-tuning: $92.28\%$ (Kém hơn $5.73\%$) | **ĐÚNG** | Fine-tuning cho phép toàn bộ 12 tầng Transformer tinh chỉnh trọng số Attention thích ứng riêng với bài toán cảm xúc. |
| **3** | Giảm dữ liệu về 10%, bên nào giảm mạnh hơn? | Model Scratch giảm mạnh hơn nhiều | Scratch giảm $-24.30\%$; Pretrained chỉ giảm $-6.08\%$ | **ĐÚNG** | Pretrained BERT có sẵn tri thức ngôn ngữ nên không bị quá khớp khi số mẫu giảm sâu về 140 bài review. |

---

## 4. Phân tích Hiện tượng Hiệu chuẩn Độ tự tin ($\text{Confidence} \neq \text{Correctness}$)

- **Quan sát thực nghiệm:** Trong nhóm mẫu có độ tự tin cao ($C \ge 0.90$), vẫn tồn tại $1.67\%$ trường hợp sai sót nghiêm trọng. Ngược lại, nhóm có độ tin cậy thấp ($C \in [0.50, 0.70)$) vẫn đạt độ chính xác thực nghiệm $69.05\%$.
- **Bản chất toán học:** Hàm Softmax có xu hướng kéo dãn khoảng cách cực đoan khi các giá trị logit chênh lệch, dẫn đến hiện tượng quá tự tin (Overconfidence Miscalibration). Độ tự tin đầu ra của mạng nơ-ron phản ánh biên độ số học của logit, không đồng nghĩa với xác suất đúng khách quan. Trong các bài toán triển khai thực tế, cần áp dụng kỹ thuật hiệu chuẩn nhiệt độ (Temperature Scaling) hoặc kiểm tra độ bất định (Uncertainty Estimation).

---

## 5. Tuyên bố sử dụng AI (AI Usage Statement)

Căn cứ theo quy định tại **Mục 24. AI Usage**, bản tuyên bố dưới đây khẳng định tính trung thực học thuật:

```yaml
AI Usage Statement:
  Tool: Antigravity IDE (DeepMind Agentic Assistant)
  Purpose:
    - Hỗ trợ xây dựng khung mã nguồn script sinh biểu đồ Matplotlib độ phân giải cao (generate_w8_plots.py).
    - Hỗ trợ định dạng cấu trúc bảng Markdown và rà soát tính hợp lệ của hai Jupyter Notebooks.
  Generated:
    - Khung cấu trúc vẽ biểu đồ Data Efficiency Curve và Reliability Diagram.
    - Bảng thống kê thông số tái hiện thực nghiệm (Reproducibility Table).
  Modified:
    - Sinh viên loại bỏ toàn bộ các comment code trong mọi tệp mã nguồn python và notebook theo quy chuẩn nghiêm ngặt.
    - Cấu hình tải tokenizer và config ở chế độ local_files_only để đảm bảo chạy hoàn toàn offline không phụ thuộc mạng.
  Verified:
    - Sinh viên trực tiếp giải chi tiết bài toán tính MLM Loss và phân tích Tokenization tại calculations.md.
    - Sinh viên tự biện luận các giả thuyết khoa học tại prediction.md và phân tích định tính tại error_analysis.md.
    - Kiểm tra tính toàn vẹn 100% của toàn bộ 12 tệp tin deliverables.
```

