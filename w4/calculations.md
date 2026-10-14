# LAB 04 — CALCULATIONS: PREDICTION, LOGITS & METRICS



## Bài 1 — Binary Classification: Logits, Sigmoid & Xác suất có điều kiện

### 1.1. Đề bài
Cho vector đặc trưng đầu vào của một văn bản:
$$x = \begin{bmatrix} 1 \\ 2 \\ 0 \end{bmatrix}$$
Vector trọng số của mô hình Logistic Regression:
$$w = \begin{bmatrix} 0.5 \\ -0.2 \\ 0.4 \end{bmatrix}$$
Hệ số chệch (bias / intercept):
$$b = 0.1$$

Yêu cầu:
1. Tính giá trị logit: $z = w^T x + b$.
2. Áp dụng hàm kích hoạt Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$ để tính xác suất dự đoán $P(y = 1 \mid x)$.

---

### 1.2. Các bước giải chi tiết

#### Bước 1: Tính tích vô hướng và giá trị logit $z$
Logit $z$ là tổ hợp tuyến tính của các đặc trưng đầu vào và trọng số tương ứng:
$$z = w^T x + b = \sum_{j=1}^d w_j x_j + b$$
Thay các giá trị số vào:
$$z = (0.5 \times 1) + (-0.2 \times 2) + (0.4 \times 0) + 0.1$$
$$z = 0.5 - 0.4 + 0 + 0.1 = 0.2$$

#### Bước 2: Áp dụng hàm Sigmoid tính xác suất $P(y = 1 \mid x)$
Hàm Sigmoid ánh xạ giá trị thực $z \in (-\infty, +\infty)$ về khoảng xác suất $(0, 1)$:
$$\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{1}{1 + e^{-0.2}}$$
Ta có:
$$e^{-0.2} \approx 0.81873075$$
Do đó:
$$\sigma(0.2) = \frac{1}{1 + 0.81873075} = \frac{1}{1.81873075} \approx 0.549834$$

#### Kết luận:
- Logit: $z = 0.2$
- Xác suất thuộc lớp Positive:
  $$P(y = 1 \mid x) = \sigma(z) \approx 0.5498 \quad (54.98\%)$$
- Xác suất thuộc lớp Negative:
  $$P(y = 0 \mid x) = 1 - P(y = 1 \mid x) \approx 0.4502 \quad (45.02\%)$$

---

## Bài 2 — Prediction Threshold & Đánh đổi Precision - Recall

### 2.1. Đề bài
Giả sử mô hình đưa ra xác suất dự đoán:
$$P(y = 1 \mid x) = 0.72$$

Yêu cầu:
1. Với ngưỡng quyết định $\tau = 0.5$, mô hình dự đoán văn bản thuộc class nào?
2. Với ngưỡng quyết định $\tau = 0.8$, mô hình dự đoán văn bản thuộc class nào?
3. Trả lời: Thay đổi threshold $\tau$ có thể làm Precision và Recall thay đổi như thế nào?

---

### 2.2. Lời giải chi tiết

#### Quy tắc phân loại theo ngưỡng $\tau$:
$$\hat{y} = \begin{cases} 1 & \text{nếu } P(y = 1 \mid x) \ge \tau \\ 0 & \text{nếu } P(y = 1 \mid x) < \tau \end{cases}$$

#### 1. Với ngưỡng chuẩn $\tau = 0.5$:
Vì $P(y = 1 \mid x) = 0.72 \ge 0.5$, mô hình dự đoán:
$$\hat{y} = 1 \quad (\text{Class Positive})$$

#### 2. Với ngưỡng khắt khe $\tau = 0.8$:
Vì $P(y = 1 \mid x) = 0.72 < 0.8$, mô hình dự đoán:
$$\hat{y} = 0 \quad (\text{Class Negative})$$

#### 3. Phân tích ảnh hưởng của Threshold tới Precision và Recall:
Ngưỡng phân loại $\tau$ là siêu tham số quyết định mức độ "thận trọng" của mô hình:

* **Khi tăng ngưỡng $\tau$ (ví dụ từ $0.5 \to 0.8$):**
  - Mô hình chỉ dự đoán là Positive khi độ tự tin rất cao.
  - Số lượng mẫu được dự đoán là Positive ($\text{TP} + \text{FP}$) giảm xuống.
  - Số lượng cảnh báo giả ($\text{FP}$) giảm mạnh $\implies$ **Precision tăng**.
  - Tuy nhiên, nhiều mẫu thực sự là Positive sẽ bị bỏ sót, làm tăng số lượng bỏ sót ($\text{FN}$) $\implies$ **Recall giảm**.

* **Khi giảm ngưỡng $\tau$ (ví dụ từ $0.5 \to 0.2$):**
  - Mô hình trở nên "dễ dãi", cố gắng bắt trọn mọi trường hợp nghi ngờ là Positive.
  - Số lượng bỏ sót ($\text{FN}$) giảm mạnh $\implies$ **Recall tăng**.
  - Nhưng mô hình chấp nhận phân loại nhầm nhiều mẫu Negative thành Positive, làm số cảnh báo giả ($\text{FP}$) tăng vọt $\implies$ **Precision giảm**.

$$\text{Tăng } \tau \implies \text{Precision } \uparrow, \quad \text{Recall } \downarrow$$
$$\text{Giảm } \tau \implies \text{Precision } \downarrow, \quad \text{Recall } \uparrow$$

Đây chính là sự đánh đổi kinh điển **Precision-Recall Trade-off**. Trong thực tế, ngưỡng $\tau$ được lựa chọn dựa trên chi phí sai lầm (cost of error) của từng bài toán cụ thể.

---

## Bài 3 — Confusion Matrix & Đánh giá phân loại Spam

### 3.1. Đề bài
Cho ma trận nhầm lẫn (Confusion Matrix) thực tế của một mô hình:

| | Predicted Positive | Predicted Negative | Tổng thực tế |
|:---|:---:|:---:|:---:|
| **Actual Positive** | $\text{TP} = 80$ | $\text{FN} = 20$ | $100$ |
| **Actual Negative** | $\text{FP} = 10$ | $\text{TN} = 90$ | $100$ |
| **Tổng dự đoán** | $90$ | $110$ | $N = 200$ |

Yêu cầu:
1. Tính Precision, Recall và $F_1$-score.
2. Nếu đây là bài toán phát hiện thư rác (Spam Detection với Positive = Spam, Negative = Ham/Hợp lệ), phân tích ý nghĩa và hệ quả thực tiễn của False Positive ($\text{FP}$) và False Negative ($\text{FN}$).

---

### 3.2. Lời giải chi tiết

#### 1. Tính toán các độ đo:

* **Precision (Độ chính xác dự đoán dương tính):**
  Tỷ lệ các mẫu thực sự là Positive trong số những mẫu được mô hình dự đoán là Positive:
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{80}{80 + 10} = \frac{80}{90} \approx 0.8889 \quad (88.89\%)$$

* **Recall (Độ nhạy / Độ bao phủ):**
  Tỷ lệ các mẫu Positive thực tế được mô hình phát hiện thành công:
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{80}{80 + 20} = \frac{80}{100} = 0.8000 \quad (80.00\%)$$

* **$F_1$-Score (Trung bình điều hòa giữa Precision và Recall):**
  $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \times \frac{\frac{8}{9} \times \frac{4}{5}}{\frac{8}{9} + \frac{4}{5}} = 2 \times \frac{\frac{32}{45}}{\frac{76}{45}} = \frac{64}{76} = \frac{16}{19} \approx 0.8421 \quad (84.21\%)$$

* **Accuracy (Độ chính xác tổng thể):**
  $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} = \frac{80 + 90}{200} = \frac{170}{200} = 0.8500 \quad (85.00\%)$$

#### 2. Ý nghĩa thực tiễn trong bài toán Spam Detection:
Quy ước: Positive = Spam (thư rác), Negative = Ham (thư hợp lệ, quan trọng).

* **False Positive ($\text{FP} = 10$):**
  - *Định nghĩa:* Một email hợp lệ (Ham) bị mô hình phân loại nhầm thành Spam và tự động chuyển vào thư mục rác (Junk/Spam folder).
  - *Hậu quả thực tế:* **Cực kỳ nghiêm trọng.** Người dùng có thể bỏ lỡ thư mời phỏng vấn xin việc, thông báo trúng tuyển đại học, hóa đơn ngân hàng, hoặc hợp đồng kinh doanh khẩn cấp. Tổn thất vô hình và hữu hình của một vụ FP là rất lớn, làm mất niềm tin của người dùng vào hệ thống email.
* **False Negative ($\text{FN} = 20$):**
  - *Định nghĩa:* Một email rác (Spam) bị mô hình bỏ lọt và xuất hiện bình thường trong hộp thư đến (Inbox).
  - *Hậu quả thực tế:* **Tương đối nhẹ.** Người dùng cảm thấy hơi khó chịu hoặc mất vài giây để bấm nút xóa/chuyển vào thùng rác. Không gây mất mát dữ liệu quan trọng.

* **Kết luận thiết kế hệ thống:**
  Trong bài toán lọc thư rác, **chi phí của $\text{FP}$ cao hơn gấp nhiều lần chi phí của $\text{FN}$** ($\text{Cost}(\text{FP}) \gg \text{Cost}(\text{FN})$). Do đó, kỹ sư NLP luôn phải cấu hình ngưỡng quyết định $\tau$ cao để đạt **Precision tối đa (ví dụ $> 99\%$)**, chấp nhận Recall thấp hơn một chút (để sót một vài email spam) chứ tuyệt đối không được để email quan trọng bị gắn nhãn nhầm thành spam.

---

## Bài 4 — Bẫy độ chính xác (The Accuracy Trap) & Dữ liệu mất cân bằng

### 4.1. Đề bài
Một tập dữ liệu kiểm thử gồm $1000$ mẫu văn bản với phân phối lớp mất cân bằng nghiêm trọng:
- $950$ mẫu Negative ($95\%$)
- $50$ mẫu Positive ($5\%$)

Một mô hình phân loại tầm thường (trivial / majority predictor) đưa ra quyết định:
$$\text{Mọi sample } x_i \longrightarrow \text{Negative}$$
Độ chính xác (Accuracy) đo được là:
$$\text{Accuracy} = \frac{950}{1000} = 95.0\%$$

Yêu cầu:
1. Mô hình này có thực sự tốt không?
2. Chỉ ra các metric chuyên biệt cần xem xét thay thế để vạch trần bản chất của mô hình.

---

### 4.2. Lời giải chi tiết

#### 1. Đánh giá mô hình:
Mô hình này **hoàn toàn vô giá trị và vô dụng trong thực tiễn (useless dummy baseline)**:
- Lập ma trận nhầm lẫn của mô hình:
  - $\text{TP} = 0$ (không phát hiện được bất kỳ mẫu Positive nào)
  - $\text{FN} = 50$ (bỏ sót toàn bộ $100\%$ mẫu Positive)
  - $\text{FP} = 0$
  - $\text{TN} = 950$
- Mặc dù Accuracy đạt con số ấn tượng $95\%$, mô hình thực chất không hề học được bất kỳ tri thức, đặc trưng ngôn ngữ hay ranh giới phân loại nào. Nó chỉ đơn giản là đoán theo lớp chiếm đa số.
- Nếu đây là bài toán phát hiện giao dịch gian lận (Fraud Detection), ung thư trong bệnh án y tế, hay mã độc trong văn bản, mô hình này sẽ gây ra thảm họa vì **tỷ lệ bỏ sót hiểm họa là $100\%$**.

#### 2. Các chỉ số chuyên biệt cần sử dụng thay thế:

1. **Recall của lớp Positive (Minority Class Recall):**
   $$\text{Recall}_{\text{pos}} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{0}{0 + 50} = 0.0\%$$
   Chỉ số này phản ánh ngay lập tức sự thất bại hoàn toàn của mô hình trong việc tìm kiếm lớp mục tiêu.

2. **Precision của lớp Positive:**
   $$\text{Precision}_{\text{pos}} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{0}{0 + 0} \implies \text{Không xác định (hoặc quy ước bằng 0)}$$

3. **$F_1$-Score của lớp Positive:**
   $$F_1(\text{pos}) = 0.0$$

4. **Macro-averaged $F_1$ Score:**
   Tính trung bình cộng $F_1$ của từng lớp, đối xử bình đẳng với mọi lớp bất kể số lượng mẫu:
   - $F_1(\text{neg}) = \frac{2 \times 950}{2 \times 950 + 0 + 0} = 1.0$
   - $F_1(\text{pos}) = 0.0$
   $$\text{Macro } F_1 = \frac{F_1(\text{neg}) + F_1(\text{pos})}{2} = \frac{1.0 + 0.0}{2} = 0.50 \quad (50.0\%)$$
   Con số $50\%$ đã vạch trần mô hình không hề tốt hơn một phép đoán ngẫu nhiên.

5. **Balanced Accuracy (Độ chính xác cân bằng):**
   Trung bình cộng của Recall từng lớp:
   $$\text{Balanced Accuracy} = \frac{\text{TPR} + \text{TNR}}{2} = \frac{\frac{0}{50} + \frac{950}{950}}{2} = \frac{0.0 + 1.0}{2} = 0.50 \quad (50.0\%)$$

6. **Đường cong PR-AUC (Precision-Recall Area Under Curve):**
   Với dữ liệu mất cân bằng nặng, PR-AUC là thước đo chuẩn mực nhất để đánh giá năng lực phân loại của mô hình trên lớp thiểu số trên toàn dải ngưỡng $\tau$.
