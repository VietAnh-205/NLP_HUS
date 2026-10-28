# LAB 06 — Giả thuyết và Dự đoán trước thực nghiệm (Prediction)


---

## 1. Prediction 1 — Tác động của Dot Product lớn hơn tới Attention Weight

### Câu hỏi:
Nếu một key có tích vô hướng (Dot Product) lớn hơn các key khác, attention weight của nó sẽ như thế nào?

- **Prediction:**
  Key có tích vô hướng lớn hơn sẽ nhận được attention weight lớn hơn các key còn lại theo quy luật **khuếch đại phi tuyến tính (Super-linear / Exponential magnification)**. Trọng số của nó sẽ chiếm tỷ lệ áp đảo trong phân phối xác suất và tạo ra sự chênh lệch vượt xa tỷ lệ chênh lệch điểm số thô ban đầu.
- **Reason:**
  1. Trọng số attention $\alpha_i$ được tính thông qua hàm chuẩn hóa Softmax:
     $$\alpha_i = \frac{\exp(s_i)}{\sum_{j=1}^T \exp(s_j)}$$
  2. Do hàm số mũ $f(x) = \exp(x)$ là hàm đồng biến nghiêm ngặt ($f'(x) > 0$) và có độ dốc tăng theo hàm mũ, đạo hàm cấp hai $f''(x) = \exp(x) > 0$ là hàm lồi.
  3. Khi một điểm số $s_k$ lớn hơn các điểm số khác một khoảng cách $\Delta > 0$, tỷ số giữa thành phần số mũ của nó so với các thành phần khác sẽ tăng theo hệ số $\exp(\Delta)$. Do đó, hàm Softmax không chỉ giữ nguyên trật tự sắp xếp của điểm số mà còn kéo giãn khoảng cách, tạo ra sự phân cực mạnh mẽ về trọng số.
- **Confidence:** **99%**

---

## 2. Prediction 2 — Trọng số Attention khi tất cả Scores đồng nhất

### Câu hỏi:
Nếu tất cả các scores đều giống nhau: $s = [1, 1, 1, 1]$, attention weights sẽ như thế nào?

- **Prediction:**
  Vector attention weights sẽ tạo thành một **phân phối đều tuyệt đối (Uniform Distribution)**:
  $$\alpha = [0.25, 0.25, 0.25, 0.25]$$
  Mỗi token nhận chính xác $\frac{1}{T} = \frac{1}{4} = 25\%$ trọng số, và context vector thu được sẽ trở thành giá trị trung bình cộng số học của tất cả các vector value:
  $$c = \frac{1}{4} \sum_{i=1}^4 v_i$$
- **Reason:**
  1. Giả sử $s_1 = s_2 = \dots = s_T = C$ (với $C = 1, T = 4$).
  2. Áp dụng công thức Softmax:
     $$\alpha_i = \frac{\exp(C)}{\sum_{j=1}^4 \exp(C)} = \frac{\exp(C)}{4 \cdot \exp(C)} = \frac{1}{4} = 0.25, \quad \forall i \in \{1, 2, 3, 4\}$$
  3. Giá trị của hằng số $C$ hoàn toàn bị triệt tiêu ở cả tử số và mẫu số. Khi mô hình không tìm thấy bất kỳ sự tương đồng đặc trưng nào nổi trội giữa Query và các Keys, Attention tự động suy biến về mô hình túi từ véc-tơ trung bình (Average Bag-of-Words / Average Word2Vec).
- **Confidence:** **100%**

---

## 3. Prediction 3 — Trọng số Attention khi xuất hiện Score ngoại lai cực lớn

### Câu hỏi:
Nếu một score lớn hơn rất nhiều so với các score còn lại: $s = [1, 1, 10]$, attention sẽ tập trung như thế nào?

- **Prediction:**
  Phân phối attention sẽ bị **co cụm cực đoan (Extreme Concentration)**, tiệm cận một vector One-hot tại vị trí phần tử có score bằng 10. Trọng số của phần tử này sẽ xấp xỉ **$99.98\%$**, trong khi hai phần tử còn lại bị triệt tiêu xuống dưới $0.01\%$:
  $$\alpha \approx [0.0001, 0.0001, 0.9998]$$
  Context vector $c$ lúc này gần như trùng khớp hoàn toàn với vector giá trị $v_3$ ($c \approx v_3$).
- **Reason:**
  1. Lũy thừa của các thành phần:
     - $\exp(1) \approx 2.7183$
     - $\exp(1) \approx 2.7183$
     - $\exp(10) \approx 22026.4658$
  2. Tổng mẫu số:
     $$\sum_{j=1}^3 \exp(s_j) = 2.7183 + 2.7183 + 22026.4658 = 22031.9024$$
  3. Tính toán trọng số cụ thể:
     $$\alpha_1 = \alpha_2 = \frac{2.7183}{22031.9024} \approx 0.000123 \approx 0.012\%$$
     $$\alpha_3 = \frac{22026.4658}{22031.9024} \approx 0.999753 \approx 99.98\%$$
  4. Cơ chế này tương đương với phép chọn cứng $\text{argmax}$, biến Softmax thành một bộ lọc chọn lọc tuyệt đối (Hard-routing selector).
- **Confidence:** **99%**

---

## 4. Prediction 4 — Tăng Sequence Length từ 10 tokens lên 1000 tokens

### Câu hỏi:
Nếu độ dài sequence tăng từ $10 \text{ tokens} \to 1000 \text{ tokens}$, chi phí tính attention sẽ thay đổi như thế nào?

- **Prediction:**
  Chi phí tính toán số phép tính (FLOPs) và dung lượng bộ nhớ RAM/VRAM để lưu trữ ma trận Attention sẽ tăng **gấp $10,000$ lần (1 vạn lần)**, tuân theo quy luật tăng trưởng bậc hai $\mathcal{O}(T^2)$.
  Cụ thể:
  - Khi $T=10$: Ma trận Attention có $10 \times 10 = 100$ phần tử ($0.4 \text{ KB}$).
  - Khi $T=1000$: Ma trận Attention có $1000 \times 1000 = 1,000,000$ phần tử ($3.81 \text{ MB}$ cho 1 ma trận đơn và $\sim 550 \text{ MB}$ cho mô hình Transformer nhiều tầng/nhiều đầu).
- **Reason:**
  1. Trong cơ chế Self-Attention, mỗi token đóng vai trò là một Query cần phải tính tích vô hướng với tất cả $T$ token khác đóng vai trò là Keys.
  2. Phép nhân ma trận $Q K^T$ với $Q \in \mathbb{R}^{T \times d_k}$ và $K^T \in \mathbb{R}^{d_k \times T}$ đòi hỏi:
     $$\text{FLOPs} = 2 \cdot T^2 \cdot d_k$$
  3. Ma trận Attention Weights $A = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)$ có kích thước cố định $T \times T$.
  4. Do độ dài chuỗi $T$ tăng từ $10 \to 1000$ (tăng hệ số $k = \frac{1000}{10} = 100$ lần), số lượng phần tử của ma trận $A$ sẽ tăng theo hệ số $k^2 = 100^2 = 10,000$ lần.
- **Confidence:** **100%**

---

## 5. Prediction 5 — Attention có luôn chọn đúng token quan trọng theo suy luận con người?

### Câu hỏi:
Attention có luôn chọn đúng token mà con người cho là quan trọng nhất hay không?

- **Prediction:**
  **Không.** Attention không nhất thiết luôn tập trung vào các từ mang tính logic hoặc ngữ pháp trọng yếu mà con người kỳ vọng.
  Nhiều trường hợp Attention sẽ phân bổ trọng số cao vào các dấu câu (dấu chấm, dấu phẩy), từ nối, mạo từ phổ biến (*"the", "a"*), hoặc phân tán mỏng trên toàn câu mà vẫn đưa ra kết quả phân loại chính xác.
- **Reason:**
  1. **Bản chất thống kê tối ưu hóa hàm mất mát (Statistical Loss Minimization):** Attention được tối ưu hóa để cực tiểu hóa hàm mất mát phân loại (Cross-Entropy Loss), không được huấn luyện theo các quy tắc ngôn ngữ học hay ngữ pháp của con người.
  2. **Dấu câu như điểm neo thông tin (Attention Sinks):** Nhiều nghiên cứu khoa học (Clark et al., 2019; Xiao et al., 2023) đã chứng minh các token như dấu chấm `.` hay dấu phẩy `,` thường đóng vai trò là "bể chứa attention" (Attention Sinks), nơi mô hình dồn phần trọng số dư thừa khi không cần tích lũy thêm ngữ cảnh.
  3. **Tương quan không đồng nghĩa với suy luận nhân quả (Attention is NOT Explanation):** Trọng số attention chỉ phản ánh tương quan tích vô hướng trong không gian ẩn của lớp đó, không phản ánh toàn bộ chu trình suy luận của mạng nơ-ron (Jain & Wallace, 2019).
- **Confidence:** **95%**
