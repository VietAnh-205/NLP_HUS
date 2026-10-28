# LAB 06 — Báo cáo Tính toán Giải tích: Attention Mechanism & Self-Attention


---

## 1. Bài 1 — Attention Scores (Mục 8)

### Đề bài:
Cho vector truy vấn (Query) $q \in \mathbb{R}^2$ và ma trận các khóa (Keys) $K \in \mathbb{R}^{3 \times 2}$:
$$q = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \quad K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}$$
Trong đó các vector khóa tương ứng là các hàng của $K$:
$$k_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad k_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad k_3 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

### Yêu cầu:
Tính điểm tương đồng tích vô hướng (Dot-product score):
$$s_i = q^T k_i \quad (i = 1, 2, 3)$$
và viết toàn bộ vector điểm số $s$.

### Lời giải chi tiết:

1. **Tính điểm số $s_1$ giữa Query $q$ và Key $k_1$:**
   $$s_1 = q^T k_1 = \begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = 1 \cdot 1 + 2 \cdot 0 = 1 + 0 = 1$$

2. **Tính điểm số $s_2$ giữa Query $q$ và Key $k_2$:**
   $$s_2 = q^T k_2 = \begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = 1 \cdot 0 + 2 \cdot 1 = 0 + 2 = 2$$

3. **Tính điểm số $s_3$ giữa Query $q$ và Key $k_3$:**
   $$s_3 = q^T k_3 = \begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = 1 \cdot 1 + 2 \cdot 1 = 1 + 2 = 3$$

Dưới dạng phép nhân ma trận:
$$s = K q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$$

**Kết luận:** Vector điểm số thu được là:
$$s = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$$

---

## 2. Bài 2 — Softmax Attention Weights (Mục 9)

### Đề bài:
Cho vector điểm số $s = [1, 2, 3]^T$. Tính vector trọng số tập trung (Attention weights) $\alpha = \text{softmax}(s)$.

### Công thức:
$$\alpha_i = \frac{\exp(s_i)}{\sum_{j=1}^3 \exp(s_j)}, \quad \forall i \in \{1, 2, 3\}$$

### Lời giải chi tiết:

1. **Tính lũy thừa tự nhiên $\exp(s_i)$ cho từng thành phần:**
   - $\exp(s_1) = \exp(1) \approx 2.71828$
   - $\exp(s_2) = \exp(2) \approx 7.38906$
   - $\exp(s_3) = \exp(3) \approx 20.08554$

2. **Tính tổng chuẩn hóa (Mẫu số):**
   $$\sum_{j=1}^3 \exp(s_j) = 2.71828 + 7.38906 + 20.08554 = 30.19288$$

3. **Tính trọng số $\alpha_i$ cho từng phần tử:**
   - $\alpha_1 = \frac{2.71828}{30.19288} \approx 0.09003 \approx 9.00\%$
   - $\alpha_2 = \frac{7.38906}{30.19288} \approx 0.24473 \approx 24.47\%$
   - $\alpha_3 = \frac{20.08554}{30.19288} \approx 0.66524 \approx 66.52\%$

4. **Kiểm tra điều kiện tổng xác suất:**
   $$\sum_{i=1}^3 \alpha_i = 0.09003 + 0.24473 + 0.66524 = 1.00000 \quad (\text{Thỏa mãn chuẩn hóa})$$

**Kết luận:** Vector trọng số attention là:
$$\alpha = \begin{bmatrix} 0.0900 \\ 0.2447 \\ 0.6652 \end{bmatrix}$$

### Trả lời câu hỏi: Vì sao phần tử có score lớn nhất nhận weight lớn nhất?
1. **Tính đơn điệu của hàm số mũ:** Hàm $f(x) = \exp(x)$ là một hàm đồng biến tuyệt đối trên $\mathbb{R}$ ($f'(x) = \exp(x) > 0$). Do đó, nếu $s_a > s_b$ thì chắc chắn $\exp(s_a) > \exp(s_b)$, kéo theo $\alpha_a > \alpha_b$.
2. **Cơ chế khuếch đại phi tuyến (Non-linear magnification):** Hàm mũ làm tăng khoảng cách tương đối giữa các giá trị lớn và nhỏ. Cụ thể:
   - Điểm $s_3 = 3$ chỉ gấp $1.5$ lần $s_2 = 2$ và gấp $3$ lần $s_1 = 1$.
   - Tuy nhiên, sau hàm mũ, $\exp(3) \approx 20.09$ gấp tới $2.72$ lần $\exp(2) \approx 7.39$ và gấp $7.39$ lần $\exp(1) \approx 2.72$.
   - Nhờ đó, trọng số $\alpha_3 = 66.52\%$ chiếm ưu thế áp đảo so với tổng hai trọng số còn lại ($33.48\%$). Đây chính là cơ chế giúp mô hình tập trung sắc bén (sharp attention) vào token phù hợp nhất.

---

## 3. Bài 3 — Context Vector (Mục 10)

### Đề bài:
Cho vector trọng số $\alpha = [0.1, 0.2, 0.7]^T$ và các vector giá trị (Values) $v_1, v_2, v_3 \in \mathbb{R}^2$:
$$v_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad v_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad v_3 = \begin{bmatrix} 2 \\ 2 \end{bmatrix}$$

### Yêu cầu:
Tính vector ngữ cảnh (Context vector):
$$c = \sum_{i=1}^3 \alpha_i v_i = 0.1 v_1 + 0.2 v_2 + 0.7 v_3$$
và giải thích vector nào đóng góp nhiều nhất vào $c$.

### Lời giải chi tiết:

1. **Tính các thành phần có trọng số:**
   $$0.1 v_1 = 0.1 \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0.1 \\ 0.0 \end{bmatrix}$$
   $$0.2 v_2 = 0.2 \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0.0 \\ 0.2 \end{bmatrix}$$
   $$0.7 v_3 = 0.7 \begin{bmatrix} 2 \\ 2 \end{bmatrix} = \begin{bmatrix} 1.4 \\ 1.4 \end{bmatrix}$$

2. **Cộng các thành phần:**
   $$c = \begin{bmatrix} 0.1 \\ 0.0 \end{bmatrix} + \begin{bmatrix} 0.0 \\ 0.2 \end{bmatrix} + \begin{bmatrix} 1.4 \\ 1.4 \end{bmatrix} = \begin{bmatrix} 0.1 + 0.0 + 1.4 \\ 0.0 + 0.2 + 1.4 \end{bmatrix} = \begin{bmatrix} 1.5 \\ 1.6 \end{bmatrix}$$

**Kết luận:** Vector ngữ cảnh thu được là:
$$c = \begin{bmatrix} 1.5 \\ 1.6 \end{bmatrix}$$

### Giải thích: Value nào đóng góp nhiều nhất vào context vector?
- Vector $v_3$ đóng góp nhiều nhất vào vector ngữ cảnh vì nó nhận trọng số lớn nhất ($\alpha_3 = 0.7 = 70\%$).
- Về mặt giá trị số học, $v_3$ đóng góp tới $[1.4, 1.4]^T$, chiếm $\frac{1.4}{1.5} \approx 93.3\%$ thành phần thứ nhất và $\frac{1.4}{1.6} = 87.5\%$ thành phần thứ hai của vector ngữ cảnh $c$.
- Trực quan: Context vector $c$ là một điểm trong bao lồi (convex hull) của tập các vector $\{v_1, v_2, v_3\}$. Vì trọng số của $v_3$ chiếm $70\%$, vị trí hình học của $c$ bị kéo mạnh mẽ nhất về phía vector $v_3$.

---

## 4. Bài 4 — Toàn bộ quy trình tính toán Attention (Mục 11)

### Đề bài:
Cho:
- Query: $q = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$
- Keys: $k_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad k_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad k_3 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$
- Values: $v_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad v_2 = \begin{bmatrix} 0 \\ 2 \end{bmatrix}, \quad v_3 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$

Thực hiện toàn bộ quy trình:
$$\text{Query } q \longrightarrow \text{Scores } s \longrightarrow \text{Softmax } \alpha \longrightarrow \text{Context Vector } c$$

### Lời giải chi tiết:

#### Bước 1: Tính toán Attention Scores ($s_i = q^T k_i$)
- $s_1 = q^T k_1 = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = 1 \cdot 1 + 0 \cdot 0 = 1$
- $s_2 = q^T k_2 = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = 1 \cdot 0 + 0 \cdot 1 = 0$
- $s_3 = q^T k_3 = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = 1 \cdot 1 + 0 \cdot 1 = 1$

Vector điểm số:
$$s = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}$$

#### Bước 2: Chuẩn hóa Softmax tính Attention Weights ($\alpha_i$)
- $\exp(s_1) = \exp(1) \approx 2.71828$
- $\exp(s_2) = \exp(0) = 1.00000$
- $\exp(s_3) = \exp(1) \approx 2.71828$

Tổng mẫu số:
$$\sum_{j=1}^3 \exp(s_j) = 2.71828 + 1.00000 + 2.71828 = 6.43656$$

Các trọng số attention:
- $\alpha_1 = \frac{2.71828}{6.43656} \approx 0.42231$
- $\alpha_2 = \frac{1.00000}{6.43656} \approx 0.15536$
- $\alpha_3 = \frac{2.71828}{6.43656} \approx 0.42231$

Kiểm tra: $\sum_{i=1}^3 \alpha_i = 0.42231 + 0.15536 + 0.42231 = 0.99998 \approx 1.0$.

Vector trọng số attention:
$$\alpha = \begin{bmatrix} 0.4223 \\ 0.1554 \\ 0.4223 \end{bmatrix}$$

#### Bước 3: Tính toán Context Vector ($c = \sum_{i=1}^3 \alpha_i v_i$)
$$c = 0.4223 \begin{bmatrix} 1 \\ 0 \end{bmatrix} + 0.1554 \begin{bmatrix} 0 \\ 2 \end{bmatrix} + 0.4223 \begin{bmatrix} 2 \\ 1 \end{bmatrix}$$
$$c = \begin{bmatrix} 0.42231 \\ 0 \end{bmatrix} + \begin{bmatrix} 0 \\ 0.31072 \end{bmatrix} + \begin{bmatrix} 0.84462 \\ 0.42231 \end{bmatrix}$$
$$c = \begin{bmatrix} 0.42231 + 0 + 0.84462 \\ 0 + 0.31072 + 0.42231 \end{bmatrix} = \begin{bmatrix} 1.26693 \\ 0.73303 \end{bmatrix} \approx \begin{bmatrix} 1.2669 \\ 0.7331 \end{bmatrix}$$

**Kết luận:** 
$$\text{Context Vector } c = \begin{bmatrix} 1.2669 \\ 0.7331 \end{bmatrix}$$

---

## 5. Bài tập — Scaled Dot-Product Attention (Mục 13, 14)

### Đề bài:
Cho chiều không gian vector khóa $d_k = 4 \implies \sqrt{d_k} = \sqrt{4} = 2$.  
Điểm tích vô hướng chưa chia tỉ lệ là:
$$Q K^T = \begin{bmatrix} 4 & 2 & 0 \end{bmatrix}$$

### Yêu cầu:
1. Tính scaled scores: $s_{scaled} = \frac{Q K^T}{\sqrt{d_k}}$.
2. Tính phân phối Softmax cho cả hai trường hợp (có scaling và không có scaling).
3. So sánh và giải thích: Scaling làm thay đổi độ tập trung của Softmax như thế nào?

### Lời giải chi tiết:

1. **Trường hợp 1: Không Scaling ($s = [4, 2, 0]$)**
   - $\exp(4) \approx 54.59815$
   - $\exp(2) \approx 7.38906$
   - $\exp(0) = 1.00000$
   - Tổng: $\sum = 54.59815 + 7.38906 + 1.00000 = 62.98721$
   - Trọng số Softmax:
     $$\alpha_1 = \frac{54.59815}{62.98721} \approx 0.86681 \approx 86.68\%$$
     $$\alpha_2 = \frac{7.38906}{62.98721} \approx 0.11731 \approx 11.73\%$$
     $$\alpha_3 = \frac{1.00000}{62.98721} \approx 0.01588 \approx 1.59\%$$
   $$\alpha_{unscaled} = \begin{bmatrix} 0.8668 & 0.1173 & 0.0159 \end{bmatrix}$$

2. **Trường hợp 2: Có Scaling ($d_k = 4 \implies \sqrt{d_k} = 2$)**
   $$s_{scaled} = \frac{[4, 2, 0]}{2} = \begin{bmatrix} 2 & 1 & 0 \end{bmatrix}$$
   - $\exp(2) \approx 7.38906$
   - $\exp(1) \approx 2.71828$
   - $\exp(0) = 1.00000$
   - Tổng: $\sum = 7.38906 + 2.71828 + 1.00000 = 11.10734$
   - Trọng số Softmax:
     $$\alpha_1 = \frac{7.38906}{11.10734} \approx 0.66524 \approx 66.52\%$$
     $$\alpha_2 = \frac{2.71828}{11.10734} \approx 0.24473 \approx 24.47\%$$
     $$\alpha_3 = \frac{1.00000}{11.10734} \approx 0.09003 \approx 9.00\%$$
   $$\alpha_{scaled} = \begin{bmatrix} 0.6652 & 0.2447 & 0.0900 \end{bmatrix}$$

### Bảng đối chiếu:

| Thành phần | $s_1$ | $s_2$ | $s_3$ | Trọng số $\alpha_1$ | Trọng số $\alpha_2$ | Trọng số $\alpha_3$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Chưa Scaling** | $4$ | $2$ | $0$ | **86.68%** | 11.73% | 1.59% |
| **Có Scaling ($\sqrt{4}=2$)** | $2$ | $1$ | $0$ | **66.52%** | 24.47% | 9.00% |

### Giải thích: Scaling làm thay đổi độ tập trung như thế nào?
- **Làm mềm phân phối (Softening the distribution):** Khi không có scaling, phần tử lớn nhất ($s_1=4$) áp đảo hoàn toàn với $86.68\%$, phân phối bị co cụm cực đoan (gần giống One-hot vector). Sau khi chia $\sqrt{d_k}$, trọng số của $s_1$ giảm xuống $66.52\%$, các phần tử còn lại tăng từ $11.73\% \to 24.47\%$ và $1.59\% \to 9.00\%$.
- **Nguyên nhân toán học và ý nghĩa đối với Backpropagation:** Giả sử các thành phần của $q$ và $k$ là các biến ngẫu nhiên độc lập có kỳ vọng bằng 0 và phương sai bằng 1:
  $$\mathbb{E}[q_i] = 0, \quad \text{Var}(q_i) = 1 \implies \text{Var}(q^T k) = \text{Var}\left(\sum_{i=1}^{d_k} q_i k_i\right) = d_k$$
  Khi số chiều $d_k$ lớn (ví dụ $d_k = 64$ hoặc $512$), phương sai của tích vô hướng sẽ là $d_k$, độ lệch chuẩn là $\sqrt{d_k}$. Các giá trị score sẽ trở nên rất lớn về độ lớn tuyệt đối.
  Khi đưa các giá trị quá lớn vào hàm Softmax, hàm này bị đẩy vào **vùng bão hòa (Saturation region)**, nơi đạo hàm của Softmax triệt tiêu:
  $$\frac{\partial \text{softmax}(s)_i}{\partial s_j} = \alpha_i (\delta_{ij} - \alpha_j) \approx 0$$
  Điều này gây ra hiện tượng **Vanishing Gradient** nghiêm trọng trong quá trình huấn luyện. Việc chia cho $\sqrt{d_k}$ đưa phương sai của score về lại bằng $1$, duy trì gradient dồi dào và ổn định.

---

## 6. Bài tập — Nhiệt độ (Temperature) trong Softmax (Mục 15)

### Đề bài:
Cho vector điểm số $s = [2, 1, 0]^T$. Tính phân phối Softmax có tham số nhiệt độ $T$:
$$\alpha_i(T) = \text{softmax}\left(\frac{s_i}{T}\right) = \frac{\exp(s_i / T)}{\sum_j \exp(s_j / T)}$$
với $T \in \{0.5, 1.0, 2.0\}$.

Tính thêm độ hỗn loạn thông tin (Entropy):
$$H(\alpha) = - \sum_{i=1}^3 \alpha_i \ln(\alpha_i)$$

### Lời giải chi tiết:

1. **Với $T = 0.5$ (Nhiệt độ thấp / Cool):**
   $$s' = \frac{[2, 1, 0]}{0.5} = \begin{bmatrix} 4 & 2 & 0 \end{bmatrix}$$
   - $\exp(4) \approx 54.598, \exp(2) \approx 7.389, \exp(0) = 1.0 \implies \sum = 62.987$
   - $\alpha = [0.8668, 0.1173, 0.0159]$
   - Max weight: **$0.8668$**
   - Entropy:
     $$H = - [0.8668 \ln(0.8668) + 0.1173 \ln(0.1173) + 0.0159 \ln(0.0159)] \approx 0.443 \text{ nats}$$

2. **Với $T = 1.0$ (Nhiệt độ chuẩn / Standard):**
   $$s' = \frac{[2, 1, 0]}{1.0} = \begin{bmatrix} 2 & 1 & 0 \end{bmatrix}$$
   - $\exp(2) \approx 7.389, \exp(1) \approx 2.718, \exp(0) = 1.0 \implies \sum = 11.107$
   - $\alpha = [0.6652, 0.2447, 0.0900]$
   - Max weight: **$0.6652$**
   - Entropy:
     $$H = - [0.6652 \ln(0.6652) + 0.2447 \ln(0.2447) + 0.0900 \ln(0.0900)] \approx 0.832 \text{ nats}$$

3. **Với $T = 2.0$ (Nhiệt độ cao / Warm):**
   $$s' = \frac{[2, 1, 0]}{2.0} = \begin{bmatrix} 1.0 & 0.5 & 0.0 \end{bmatrix}$$
   - $\exp(1.0) \approx 2.7183, \exp(0.5) \approx 1.6487, \exp(0.0) = 1.0000 \implies \sum = 5.3670$
   - $\alpha_1 = \frac{2.7183}{5.3670} \approx 0.5065$
   - $\alpha_2 = \frac{1.6487}{5.3670} \approx 0.3072$
   - $\alpha_3 = \frac{1.0000}{5.3670} \approx 0.1863$
   - Max weight: **$0.5065$**
   - Entropy:
     $$H = - [0.5065 \ln(0.5065) + 0.3072 \ln(0.3072) + 0.1863 \ln(0.1863)] \approx 1.018 \text{ nats}$$

### Bảng tổng hợp kết quả Temperature:

| Temperature ($T$) | Vector Softmax $\alpha$ | Max Weight ($\alpha_1$) | Entropy ($H$) | Nhận xét tính chất phân phối |
| :---: | :---: | :---: | :---: | :--- |
| **0.5** | $[0.8668, 0.1173, 0.0159]$ | **0.8668** | 0.443 | Phân phối cực kỳ tập trung, sắc bén (Sharp / Peaked), tiệm cận hàm $\text{argmax}$. |
| **1.0** | $[0.6652, 0.2447, 0.0900]$ | **0.6652** | 0.832 | Phân phối cân bằng tự nhiên giữa tập trung và lan tỏa thông tin. |
| **2.0** | $[0.5065, 0.3072, 0.1863]$ | **0.5065** | 1.018 | Phân phối phẳng (Smooth / Flattened), tiệm cận phân phối đều (Uniform), Entropy cao. |

---

## 7. Bài tập — Tính ổn định số học của Softmax (Mục 17)

### Đề bài:
Công thức tính Softmax ổn định về số học (Numerically stable softmax):
$$\text{softmax}(x_i) = \frac{\exp(x_i - \max(x))}{\sum_j \exp(x_j - \max(x))}$$
Cho vector điểm số cực lớn:
$$x = \begin{bmatrix} 1000 & 1001 & 1002 \end{bmatrix}$$

### Yêu cầu:
1. Chứng minh toán học rằng việc trừ đi đại lượng cố định $C = \max(x)$ không làm thay đổi giá trị của Softmax.
2. Tính toán kết quả cụ thể trên vector $x$ và giải thích hiện tượng tràn số (Overflow).

### Lời giải chi tiết:

#### 1. Chứng minh toán học:
Giả sử ta trừ đi một hằng số tùy ý $C \in \mathbb{R}$ từ mỗi phần tử $x_i$:
$$\frac{\exp(x_i - C)}{\sum_j \exp(x_j - C)} = \frac{\exp(x_i) \cdot \exp(-C)}{\sum_j \left(\exp(x_j) \cdot \exp(-C)\right)}$$
Do $\exp(-C)$ là một hằng số dương khác 0, ta đưa ra ngoài dấu tổng ở mẫu số:
$$= \frac{\exp(x_i) \cdot \exp(-C)}{\exp(-C) \cdot \sum_j \exp(x_j)} = \frac{\exp(x_i)}{\sum_j \exp(x_j)} = \text{softmax}(x_i) \quad (\text{đpcm})$$

#### 2. Phân tích hiện tượng Overflow và tính toán trên $x = [1000, 1001, 1002]$:
- Trong chuẩn biểu diễn số thực IEEE 754 float64 (double precision), số thực lớn nhất có thể biểu diễn xấp xỉ $\approx 1.79 \times 10^{308}$.
  - Giới hạn trên để tính hàm mũ là $\ln(1.79 \times 10^{308}) \approx 709.78$.
  - Khi tính $\exp(1000)$, giá trị này là $\approx 1.97 \times 10^{434} \gg 1.79 \times 10^{308}$, dẫn đến lỗi **Overflow $\to \infty$ (`inf`)**.
  - Kết quả $\frac{\infty}{\infty}$ sẽ sinh ra giá trị bất định `NaN` (Not a Number), làm sụp đổ toàn bộ quá trình huấn luyện mạng nơ-ron.

- **Áp dụng Numerically Stable Softmax:**
  Chọn $C = \max(x) = 1002$.
  Vector đã dịch chuyển:
  $$x' = x - 1002 = \begin{bmatrix} 1000 - 1002 & 1001 - 1002 & 1002 - 1002 \end{bmatrix} = \begin{bmatrix} -2 & -1 & 0 \end{bmatrix}$$
  Bây giờ tất cả các phần tử $x'_i \le 0$, do đó $\exp(x'_i) \in (0, 1]$, triệt tiêu hoàn toàn nguy cơ tràn số dương!

  Tính toán:
  - $\exp(-2) \approx 0.135335$
  - $\exp(-1) \approx 0.367879$
  - $\exp(0) = 1.000000$
  - Tổng: $\sum = 0.135335 + 0.367879 + 1.000000 = 1.503214$

  Các trọng số Softmax:
  - $\alpha_1 = \frac{0.135335}{1.503214} \approx 0.09003 \approx 9.00\%$
  - $\alpha_2 = \frac{0.367879}{1.503214} \approx 0.24473 \approx 24.47\%$
  - $\alpha_3 = \frac{1.000000}{1.503214} \approx 0.66524 \approx 66.52\%$

**Kết luận:** Vector Softmax thu được chính xác là:
$$\alpha = \begin{bmatrix} 0.0900 & 0.2447 & 0.6652 \end{bmatrix}$$
*(Kết quả trùng khớp tuyệt đối với Bài 2 khi các điểm số ban đầu có cùng khoảng chênh lệch).*

---

## 8. Bài tập — Self-Attention Matrix và Tương tác Nội suy (Mục 20, 21, 22)

### Đề bài:
Cho câu gồm $T=3$ tokens: `["I", "like", "cats"]`.  
Ma trận biểu diễn từ ban đầu $X \in \mathbb{R}^{3 \times 2}$:
$$X = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \quad \begin{matrix} \text{("I")} \\ \text{("like")} \\ \text{("cats")} \end{matrix}$$

Cho các ma trận trọng số chiếu tuyến tính $W_Q, W_K, W_V \in \mathbb{R}^{2 \times 2}$ ($d_k = 2 \implies \sqrt{d_k} = \sqrt{2} \approx 1.4142$):
$$W_Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad W_K = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}, \quad W_V = \begin{bmatrix} 2 & 0 \\ 1 & 1 \end{bmatrix}$$

### Yêu cầu:
1. Tính các ma trận Query $Q$, Key $K$, Value $V$.
2. Tính ma trận điểm thô $Q K^T$.
3. Tính ma trận Self-Attention $A = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{3 \times 3}$.
4. Tính ma trận biểu diễn ngữ cảnh mới $C = A V \in \mathbb{R}^{3 \times 2}$.

### Lời giải chi tiết:

#### 1. Tính các ma trận $Q, K, V$:
- **Query Matrix $Q = X W_Q$:**
  $$Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}$$

- **Key Matrix $K = X W_K$:**
  $$K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1 \cdot 1 + 0 \cdot 0 & 1 \cdot 1 + 0 \cdot 1 \\ 0 \cdot 1 + 1 \cdot 0 & 0 \cdot 1 + 1 \cdot 1 \\ 1 \cdot 1 + 1 \cdot 0 & 1 \cdot 1 + 1 \cdot 1 \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 2 \end{bmatrix}$$

- **Value Matrix $V = X W_V$:**
  $$V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 2 & 0 \\ 1 & 1 \end{bmatrix} = \begin{bmatrix} 1 \cdot 2 + 0 \cdot 1 & 1 \cdot 0 + 0 \cdot 1 \\ 0 \cdot 2 + 1 \cdot 1 & 0 \cdot 0 + 1 \cdot 1 \\ 1 \cdot 2 + 1 \cdot 1 & 1 \cdot 0 + 1 \cdot 1 \end{bmatrix} = \begin{bmatrix} 2 & 0 \\ 1 & 1 \\ 3 & 1 \end{bmatrix}$$

#### 2. Tính ma trận điểm tương tác $S_{raw} = Q K^T$:
Chuyển vị của $K$:
$$K^T = \begin{bmatrix} 1 & 0 & 1 \\ 1 & 1 & 2 \end{bmatrix}$$
Nhân ma trận $Q K^T$:
$$S_{raw} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 & 1 \\ 1 & 1 & 2 \end{bmatrix} = \begin{bmatrix} 1 \cdot 1 + 0 \cdot 1 & 1 \cdot 0 + 0 \cdot 1 & 1 \cdot 1 + 0 \cdot 2 \\ 0 \cdot 1 + 1 \cdot 1 & 0 \cdot 0 + 1 \cdot 1 & 0 \cdot 1 + 1 \cdot 2 \\ 1 \cdot 1 + 1 \cdot 1 & 1 \cdot 0 + 1 \cdot 1 & 1 \cdot 1 + 1 \cdot 2 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 1 \\ 1 & 1 & 2 \\ 2 & 1 & 3 \end{bmatrix}$$

#### 3. Tính ma trận Scaled Scores $S = \frac{Q K^T}{\sqrt{2}}$ ($\sqrt{2} \approx 1.4142$):
$$S = \begin{bmatrix} \frac{1}{1.4142} & \frac{0}{1.4142} & \frac{1}{1.4142} \\ \frac{1}{1.4142} & \frac{1}{1.4142} & \frac{2}{1.4142} \\ \frac{2}{1.4142} & \frac{1}{1.4142} & \frac{3}{1.4142} \end{bmatrix} \approx \begin{bmatrix} 0.7071 & 0.0000 & 0.7071 \\ 0.7071 & 0.7071 & 1.4142 \\ 1.4142 & 0.7071 & 2.1213 \end{bmatrix}$$

#### 4. Tính ma trận Self-Attention $A = \text{softmax}(S)$ (Softmax áp dụng theo từng hàng):
- **Hàng 1 (Token "I"):**
  Scores: $[0.7071, 0.0000, 0.7071]$
  $\exp = [2.0281, 1.0000, 2.0281] \implies \text{Tổng} = 5.0562$
  Trọng số hàng 1:
  $$A_{1,:} = \begin{bmatrix} \frac{2.0281}{5.0562} & \frac{1.0000}{5.0562} & \frac{2.0281}{5.0562} \end{bmatrix} = \begin{bmatrix} 0.4011 & 0.1978 & 0.4011 \end{bmatrix}$$

- **Hàng 2 (Token "like"):**
  Scores: $[0.7071, 0.7071, 1.4142]$
  $\exp = [2.0281, 2.0281, 4.1132] \implies \text{Tổng} = 8.1694$
  Trọng số hàng 2:
  $$A_{2,:} = \begin{bmatrix} \frac{2.0281}{8.1694} & \frac{2.0281}{8.1694} & \frac{4.1132}{8.1694} \end{bmatrix} = \begin{bmatrix} 0.2483 & 0.2483 & 0.5035 \end{bmatrix}$$

- **Hàng 3 (Token "cats"):**
  Scores: $[1.4142, 0.7071, 2.1213]$
  $\exp = [4.1132, 2.0281, 8.3420] \implies \text{Tổng} = 14.4833$
  Trọng số hàng 3:
  $$A_{3,:} = \begin{bmatrix} \frac{4.1132}{14.4833} & \frac{2.0281}{14.4833} & \frac{8.3420}{14.4833} \end{bmatrix} = \begin{bmatrix} 0.2840 & 0.1400 & 0.5760 \end{bmatrix}$$

Ma trận Self-Attention hoàn chỉnh:
$$A = \begin{bmatrix} 0.4011 & 0.1978 & 0.4011 \\ 0.2483 & 0.2483 & 0.5035 \\ 0.2840 & 0.1400 & 0.5760 \end{bmatrix}$$

#### 5. Tính ma trận biểu diễn ngữ cảnh mới $C = A V$:
$$C = \begin{bmatrix} 0.4011 & 0.1978 & 0.4011 \\ 0.2483 & 0.2483 & 0.5035 \\ 0.2840 & 0.1400 & 0.5760 \end{bmatrix} \begin{bmatrix} 2 & 0 \\ 1 & 1 \\ 3 & 1 \end{bmatrix}$$
- **Hàng 1 (Ngữ cảnh của "I"):**
  $$c_1 = 0.4011 \begin{bmatrix} 2 \\ 0 \end{bmatrix} + 0.1978 \begin{bmatrix} 1 \\ 1 \end{bmatrix} + 0.4011 \begin{bmatrix} 3 \\ 1 \end{bmatrix} = \begin{bmatrix} 0.8022 + 0.1978 + 1.2033 \\ 0 + 0.1978 + 0.4011 \end{bmatrix} = \begin{bmatrix} 2.2033 \\ 0.5989 \end{bmatrix}$$
- **Hàng 2 (Ngữ cảnh của "like"):**
  $$c_2 = 0.2483 \begin{bmatrix} 2 \\ 0 \end{bmatrix} + 0.2483 \begin{bmatrix} 1 \\ 1 \end{bmatrix} + 0.5035 \begin{bmatrix} 3 \\ 1 \end{bmatrix} = \begin{bmatrix} 0.4966 + 0.2483 + 1.5105 \\ 0 + 0.2483 + 0.5035 \end{bmatrix} = \begin{bmatrix} 2.2554 \\ 0.7518 \end{bmatrix}$$
- **Hàng 3 (Ngữ cảnh của "cats"):**
  $$c_3 = 0.2840 \begin{bmatrix} 2 \\ 0 \end{bmatrix} + 0.1400 \begin{bmatrix} 1 \\ 1 \end{bmatrix} + 0.5760 \begin{bmatrix} 3 \\ 1 \end{bmatrix} = \begin{bmatrix} 0.5680 + 0.1400 + 1.7280 \\ 0 + 0.1400 + 0.5760 \end{bmatrix} = \begin{bmatrix} 2.4360 \\ 0.7160 \end{bmatrix}$$

$$C = \begin{bmatrix} 2.2033 & 0.5989 \\ 2.2554 & 0.7518 \\ 2.4360 & 0.7160 \end{bmatrix}$$

**Ý nghĩa cốt lõi:** Sau cơ chế Self-Attention, vector biểu diễn của mỗi token không còn đứng độc lập mà đã hòa trộn thông tin ngữ nghĩa từ tất cả các token khác trong câu theo trọng số tương quan trực tiếp.

---

## 9. Bài tập — Độ phức tạp tính toán: "Attention Không Miễn Phí" (Mục 25, 29)

### Đề bài:
Ma trận Self-Attention $A$ của một chuỗi $T$ tokens có kích thước $T \times T$, dẫn đến số lượng phần tử tăng theo $\mathcal{O}(T^2)$.
So sánh số phần tử và ước lượng bộ nhớ lưu trữ ma trận Attention (với kiểu dữ liệu `float32` = 4 bytes) cho các độ dài:
$$T \in \{10, 100, 500, 1000, 10000\}$$

### Lời giải chi tiết:

1. **Công thức tính số phần tử:** $N_{elements} = T^2$
2. **Công thức tính bộ nhớ lưu trữ:**
   $$\text{Memory (Bytes)} = T^2 \times 4 \text{ bytes}$$
   $$\text{Memory (MB)} = \frac{T^2 \times 4}{1024^2} \text{ MB}, \quad \text{Memory (GB)} = \frac{T^2 \times 4}{1024^3} \text{ GB}$$

### Bảng phân tích chi phí tính toán và bộ nhớ:

| Độ dài chuỗi ($T$) | Kích thước ma trận $A$ | Số phần tử ($T^2$) | Bộ nhớ cho 1 Attention Matrix | Bộ nhớ cho 12 Layers $\times$ 12 Heads (Batch=1) | Độ tăng chi phí so với $T=10$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **10** | $10 \times 10$ | $100$ | $400 \text{ B}$ | $56.25 \text{ KB}$ | $1\times$ (Mốc cơ sở) |
| **100** | $100 \times 100$ | $10,000$ | $39.06 \text{ KB}$ | $5.49 \text{ MB}$ | $100\times$ |
| **500** | $500 \times 500$ | $250,000$ | $976.56 \text{ KB}$ | $137.33 \text{ MB}$ | $2,500\times$ |
| **1,000** | $1000 \times 1000$ | $1,000,000$ | **3.81 MB** | **549.32 MB** | $10,000\times$ |
| **10,000** | $10000 \times 10000$ | $100,000,000$ | **381.47 MB** | **53.64 GB (Tràn VRAM GPU)** | $1,000,000\times$ |

### Phân tích và Trả lời câu hỏi:
**Tại sao Self-Attention gặp phải rào cản chi phí tính toán khi chuỗi rất dài?**
1. **Sự bùng nổ bậc hai (Quadratic Bottleneck $\mathcal{O}(T^2)$):**
   - Với mạng RNN, chi phí tính toán tỉ lệ thuận tuyến tính với số bước thời gian $\mathcal{O}(T \cdot d^2)$. Khi độ dài chuỗi tăng gấp $10$ lần ($1,000 \to 10,000$), chi phí của RNN chỉ tăng $10$ lần.
   - Với Self-Attention, mỗi token phải tính điểm tương đồng với toàn bộ $T$ tokens khác trong câu. Khi $T$ tăng gấp $10$ lần, số phép tính $Q K^T$ và kích thước ma trận Softmax tăng gấp $10^2 = 100$ lần; khi $T$ tăng từ $10 \to 10,000$ ($1,000$ lần), chi phí bùng nổ tới **1,000,000 lần**!
2. **Nghẽn cổ chai bộ nhớ GPU (VRAM Memory Bottleneck):**
   - Để thực hiện thuật toán lan truyền ngược (Backpropagation), toàn bộ ma trận Attention weights $A \in \mathbb{R}^{B \times H \times T \times T}$ phải được lưu trữ trong VRAM.
   - Với chuỗi $T = 10,000$, một mô hình Transformer tiêu chuẩn (12 layers, 12 attention heads, batch size = 1) đã tiêu tốn tới **53.64 GB VRAM** chỉ để lưu riêng ma trận attention, vượt quá giới hạn của các GPU thương mại phổ biến (RTX 3090/4090 có 24 GB VRAM).
3. **Kết luận nhận thức:**
   - Attention không phải là "miễn phí". Nó đánh đổi chi phí bộ nhớ và tính toán bậc hai $\mathcal{O}(T^2)$ để đổi lấy đường truyền thông tin trực tiếp tầm xa $\mathcal{O}(1)$ và khả năng tính toán song song hoàn toàn.
   - Đây chính là động lực thúc đẩy sự ra đời của các kiến trúc tối ưu sau này như Sparse Attention, FlashAttention, Linear Attention hoặc Longformer.
