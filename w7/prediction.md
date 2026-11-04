# LAB 07 — Giả thuyết và Dự đoán trước thực nghiệm (Prediction)


---

## 1. Prediction 1 — Khả năng xử lý song song so với RNN

### Câu hỏi:
Một Transformer có thể xử lý các token trong sequence song song hơn RNN không?

- **Prediction:**
  **Có, vượt trội hoàn toàn.** Transformer có khả năng song song hóa toàn bộ chuỗi token trong cả quá trình lan truyền xuôi (Forward) và lan truyền ngược (Backward), trong khi RNN bị chặn bởi tính tuần tự nghiêm ngặt.
- **Reason:**
  1. Trong RNN, trạng thái ẩn tại bước $t$ đòi hỏi phải có trạng thái ẩn tại bước $t-1$ ($\vec{h}_t = f(\vec{x}_t, \vec{h}_{t-1})$), tạo ra một chuỗi phụ thuộc thời gian có độ sâu tính toán $\mathcal{O}(T)$. GPU không thể tính song song token thứ $t$ khi token $t-1$ chưa hoàn thành.
  2. Trong Transformer Encoder, tất cả các phép toán $Q = X W_Q, K = X W_K, V = X W_V$ và ma trận Self-Attention $S = \frac{Q K^T}{\sqrt{d_k}}$ đều là các phép nhân ma trận khối (GEMM) được phân tán đồng thời trên hàng nghìn nhân CUDA của GPU, đạt độ sâu tính toán $\mathcal{O}(1)$ về mặt thời gian.
- **Confidence:** **100%**

---

## 2. Prediction 2 — Nhận biết thứ tự khi bỏ Positional Encoding

### Câu hỏi:
Nếu bỏ positional encoding, Transformer có còn biết thứ tự token không?

- **Prediction:**
  **Không.** Nếu không có Positional Encoding, Self-Attention thuần túy hoàn toàn không có khái niệm về vị trí (Permutation Invariant đối với tập hợp token).
- **Reason:**
  1. Cơ chế Self-Attention tính điểm tương tác $s_{ij} = \frac{q_i^T k_j}{\sqrt{d_k}}$ chỉ dựa vào sự tương quan nội tại giữa hai vector embedding $x_i$ và $x_j$, không phụ thuộc vào chỉ số $i$ và $j$ trong chuỗi.
  2. Nếu hoán vị ngẫu nhiên thứ tự các hàng trong ma trận đầu vào $X$, ma trận đầu ra của khối Self-Attention chỉ bị hoán vị các hàng tương ứng mà không làm thay đổi các giá trị vector ngữ cảnh thu được.
  3. Khi áp dụng Mean Pooling để tạo vector đại diện cho toàn bộ câu, kết quả thu được sẽ hoàn toàn đồng nhất đối với mọi hoán vị của cùng một tập từ.
- **Confidence:** **99%**

---

## 3. Prediction 3 — Phân biệt hai câu hoán vị từ ("dog bites man" vs "man bites dog")

### Câu hỏi:
Nếu hai câu có cùng tập token nhưng khác thứ tự, Transformer không có positional information có thể phân biệt chúng không?

- **Prediction:**
  **Không thể phân biệt được.** Hai câu $S_1 =$ *"dog bites man"* và $S_2 =$ *"man bites dog"* sẽ tạo ra các ma trận biểu diễn là hoán vị hàng của nhau. Khi đưa qua bộ phân loại sử dụng phép gộp bất biến theo hoán vị (như Mean Pooling hay Sum Pooling), logits và nhãn dự đoán cho cả hai câu sẽ hoàn toàn trùng khớp.
- **Reason:**
  1. Cả hai câu đều có cùng tập hợp token: $\{\text{"dog"}, \text{"bites"}, \text{"man"}\}$.
  2. Khi không có Positional Encoding:
     $$x_{\text{dog}}^{(S_1)} = x_{\text{dog}}^{(S_2)}, \quad x_{\text{bites}}^{(S_1)} = x_{\text{bites}}^{(S_2)}, \quad x_{\text{man}}^{(S_1)} = x_{\text{man}}^{(S_2)}$$
  3. Vector ngữ cảnh của từng token sau Attention chỉ phụ thuộc vào tích vô hướng với hai từ còn lại. Tập hợp các vector ngữ cảnh $\{z_{\text{dog}}, z_{\text{bites}}, z_{\text{man}}\}$ là hoàn toàn giống nhau giữa $S_1$ và $S_2$.
  4. Phép tính $h_{\text{mean}} = \frac{1}{3}(z_{\text{dog}} + z_{\text{bites}} + z_{\text{man}})$ cho ra cùng một vector duy nhất. Chỉ khi cộng thêm $PE(pos)$, vector tại vị trí 1 mới khác vector tại vị trí 3.
- **Confidence:** **99%**

---

## 4. Prediction 4 — Tính đa dạng của các Attention Heads

### Câu hỏi:
Hai attention heads có nhất thiết học cùng một attention pattern không?

- **Prediction:**
  **Không nhất thiết.** Các heads khác nhau sẽ có xu hướng học các mẫu hình tương tác (Attention Patterns) khác nhau, phản ánh các mối quan hệ ngữ pháp hoặc ngữ nghĩa đa diện (chủ ngữ - vị ngữ, từ phủ định - tính từ, từ kế cận, hoặc phân tán toàn cục).
- **Reason:**
  1. Mỗi head $i$ được khởi tạo với các ma trận trọng số chiếu độc lập $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{model} \times d_k}$.
  2. Quá trình tối ưu hóa bằng Gradient Descent trong không gian nhiều chiều sẽ dẫn dắt từng head tối ưu hóa các không gian con trực giao khác nhau nhằm cực tiểu hóa hàm mất mát phân loại.
  3. Tuy nhiên, nếu không có cơ chế phạt tương quan (regularization) hoặc khi dữ liệu quá nhỏ, một số heads vẫn có thể rơi vào trạng thái bão hòa hoặc trùng lặp (head redundancy).
- **Confidence:** **90%**

---

## 5. Prediction 5 — Tác động của việc tăng số Attention Heads tới Accuracy

### Câu hỏi:
Tăng số attention heads có chắc chắn làm accuracy tăng không?

- **Prediction:**
  **Không chắc chắn.** Tăng số heads chỉ mang lại lợi ích khi số chiều của từng head $d_k = \frac{d_{model}}{h}$ đủ lớn để chứa thông tin ngữ nghĩa. Khi $d_{model}$ cố định, tăng $h$ quá mức sẽ làm $d_k$ bị thu hẹp cực độ, khiến từng head mất khả năng biểu diễn và làm hiệu năng bão hòa hoặc suy giảm.
- **Reason:**
  1. Với $d_{model} = 128$:
     - $h = 4 \implies d_k = 32$: Điểm cân bằng tối ưu giữa tính đa diện và độ sâu không gian con.
     - $h = 8 \implies d_k = 16$; $h = 16 \implies d_k = 8$: Không gian vector quá nhỏ ($8$ chiều) không đủ sức chứa các cấu trúc biểu diễn phức tạp.
  2. Nhiều heads hơn đồng nghĩa với việc phân mảnh không gian biểu diễn và tăng chi phí tính toán ghép nối, không đảm bảo tăng tuyến tính về Accuracy trên tập dữ liệu nhỏ.
- **Confidence:** **95%**

---

## 6. Prediction 6 — Tốc độ Transformer vs RNN trên mọi chiều dài chuỗi

### Câu hỏi:
Transformer có luôn nhanh hơn RNN trên mọi sequence length không?

- **Prediction:**
  **Không.** Transformer chỉ nhanh hơn RNN trên các chuỗi có độ dài ngắn đến trung bình ($T \le 512$). Trên các chuỗi cực dài ($T > 2000 - 4000$), chi phí tính toán và bộ nhớ bậc hai $\mathcal{O}(T^2)$ của Transformer sẽ vượt qua chi phí tuyến tính $\mathcal{O}(T)$ của RNN, khiến Transformer chạy chậm hơn hoặc gây lỗi Out-Of-Memory (OOM).
- **Reason:**
  1. Độ phức tạp tính toán theo $T$:
     - RNN: $\mathcal{O}(T \cdot d^2)$ (Tuyến tính theo $T$).
     - Transformer: $\mathcal{O}(T^2 \cdot d + T \cdot d^2)$ (Bậc hai theo $T$).
  2. Khi $T$ nhỏ ($T = 64, 128$), chi phí $\mathcal{O}(T^2)$ rất nhỏ so với tốc độ song song hóa phần cứng của GPU, Transformer vượt trội về thời gian thực thi (Wall-clock time).
  3. Khi $T$ rất lớn ($T \to 10,000$), đại lượng $T^2$ bùng nổ lên $10^8$ phép tính, làm nghẽn băng thông bộ nhớ và triệt tiêu toàn bộ ưu thế song song của Transformer.
- **Confidence:** **95%**

