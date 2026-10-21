# LAB 05 — CALCULATIONS: SEQUENCE MODELING, RNN & LSTM GATES


---

## Bài 1 — Một bước tính toán thuận trong RNN (One Step RNN Forward)

### 1.1. Đề bài
Cho vector nhúng đầu vào của token tại thời điểm $t$:
$$x_t = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$$
Vector trạng thái ẩn (hidden state) tại thời điểm trước đó $t-1$:
$$h_{t-1} = \begin{bmatrix} 0.5 \\ 0.2 \end{bmatrix}$$
Ma trận trọng số đầu vào - ẩn (Input-to-hidden weights):
$$W_{xh} = \begin{bmatrix} 0.5 & 0.2 \\ 0.1 & 0.4 \end{bmatrix}$$
Ma trận trọng số ẩn - ẩn (Recurrent weights / Hidden-to-hidden):
$$W_{hh} = \begin{bmatrix} 0.3 & 0.1 \\ 0.2 & 0.5 \end{bmatrix}$$
Vector hệ số chệch (bias):
$$b_h = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

Yêu cầu:
1. Tính giá trị tiền kích hoạt: $a_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h$.
2. Áp dụng hàm kích hoạt phi tuyến $\tanh$ để tính trạng thái ẩn mới: $h_t = \tanh(a_t)$.

---

### 1.2. Các bước giải chi tiết

#### Bước 1: Tính tích ma trận $W_{xh} x_t$
$$W_{xh} x_t = \begin{bmatrix} 0.5 & 0.2 \\ 0.1 & 0.4 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} (0.5 \times 1) + (0.2 \times 0) \\ (0.1 \times 1) + (0.4 \times 0) \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0.1 \end{bmatrix}$$

#### Bước 2: Tính tích ma trận $W_{hh} h_{t-1}$
$$W_{hh} h_{t-1} = \begin{bmatrix} 0.3 & 0.1 \\ 0.2 & 0.5 \end{bmatrix} \begin{bmatrix} 0.5 \\ 0.2 \end{bmatrix} = \begin{bmatrix} (0.3 \times 0.5) + (0.1 \times 0.2) \\ (0.2 \times 0.5) + (0.5 \times 0.2) \end{bmatrix} = \begin{bmatrix} 0.15 + 0.02 \\ 0.10 + 0.10 \end{bmatrix} = \begin{bmatrix} 0.17 \\ 0.20 \end{bmatrix}$$

#### Bước 3: Tính giá trị tiền kích hoạt (Pre-activation) $a_t$
$$a_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h = \begin{bmatrix} 0.5 \\ 0.1 \end{bmatrix} + \begin{bmatrix} 0.17 \\ 0.20 \end{bmatrix} + \begin{bmatrix} 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 0.67 \\ 0.30 \end{bmatrix}$$

#### Bước 4: Áp dụng hàm kích hoạt $\tanh$ tính $h_t$
Định nghĩa hàm hyperbolic tangent:
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

1. Với $z_1 = 0.67$:
   $$e^{0.67} \approx 1.954244, \quad e^{-0.67} \approx 0.511707$$
   $$\tanh(0.67) = \frac{1.954244 - 0.511707}{1.954244 + 0.511707} = \frac{1.442537}{2.465951} \approx 0.584982$$

2. Với $z_2 = 0.30$:
   $$e^{0.30} \approx 1.349859, \quad e^{-0.30} \approx 0.740818$$
   $$\tanh(0.30) = \frac{1.349859 - 0.740818}{1.349859 + 0.740818} = \frac{0.609041}{2.090677} \approx 0.291313$$

#### Kết luận:
$$a_t = \begin{bmatrix} 0.67 \\ 0.30 \end{bmatrix}, \quad h_t = \tanh(a_t) \approx \begin{bmatrix} 0.5850 \\ 0.2913 \end{bmatrix}$$

---

## Bài 2 — Luồng thông tin (Information Flow) qua các bước thời gian

### 2.1. Đề bài
Cho chuỗi câu: `"I did not like this movie"`.  
Giả sử mô hình RNN đọc tuần tự từ trái sang phải qua 6 bước thời gian ($t=1 \to 6$):
$$h_0 \xrightarrow{\text{"I"}} h_1 \xrightarrow{\text{"did"}} h_2 \xrightarrow{\text{"not"}} h_3 \xrightarrow{\text{"like"}} h_4 \xrightarrow{\text{"this"}} h_5 \xrightarrow{\text{"movie"}} h_6$$

Yêu cầu:
1. Mô tả quá trình cập nhật trạng thái ẩn sau từng bước.
2. Tại thời điểm mô hình đọc từ `"like"` ($t=4$), thông tin từ `"not"` ($t=3$) ảnh hưởng tới trạng thái ẩn $h_4$ như thế nào?
3. So sánh cơ chế này với biểu diễn Average Word2Vec ở Lab 04.

---

### 2.2. Lời giải chi tiết

#### 1. Quá trình truyền trạng thái qua các bước thời gian:
- **$t=1$ (Đọc `"I"`):** $h_1 = \tanh(W_{xh} x_{\text{"I"}} + W_{hh} h_0 + b_h)$. Khởi tạo ngữ cảnh với chủ ngữ ngôi thứ nhất.
- **$t=2$ (Đọc `"did"`):** $h_2 = \tanh(W_{xh} x_{\text{"did"}} + W_{hh} h_1 + b_h)$. Bổ sung thông tin trợ động từ thời quá khứ.
- **$t=3$ (Đọc `"not"`):** $h_3 = \tanh(W_{xh} x_{\text{"not"}} + W_{hh} h_2 + b_h)$. **Bước ngoặt ngữ nghĩa:** Trạng thái $h_3$ tích lũy tín hiệu phủ định mạnh mẽ, chuẩn bị biến đổi cực tính của các động từ tiếp theo.
- **$t=4$ (Đọc `"like"`):** $h_4 = \tanh(W_{xh} x_{\text{"like"}} + W_{hh} h_3 + b_h)$.
- **$t=5$ (Đọc `"this"`):** $h_5 = \tanh(W_{xh} x_{\text{"this"}} + W_{hh} h_4 + b_h)$.
- **$t=6$ (Đọc `"movie"`):** $h_6 = \tanh(W_{xh} x_{\text{"movie"}} + W_{hh} h_5 + b_h)$. $h_6$ là biểu diễn tổng hợp cuối cùng đưa vào tầng phân loại.

#### 2. Ảnh hưởng của `"not"` tới trạng thái ẩn $h_4$ khi đọc `"like"`:
Tại $t=4$, vector đầu vào là $x_{\text{"like"}}$ (mang tải trọng cảm xúc tích cực). Tuy nhiên, công thức cập nhật trạng thái là:
$$h_4 = \tanh(W_{xh} x_{\text{"like"}} + W_{hh} h_3 + b_h)$$
- Thành phần $W_{hh} h_3$ mang toàn bộ thông tin phủ định tích lũy từ các bước trước (đặc biệt là `"not"` ở $t=3$).
- Khi được nhân với ma trận hồi quy $W_{hh}$, vector $h_3$ đóng vai trò như một toán tử biến đổi ngữ cảnh, làm lệch hướng vector tiền kích hoạt $a_4$.
- Nhờ hàm phi tuyến $\tanh$, trạng thái $h_4$ sinh ra **không phải là trạng thái hưng phấn tích cực của từ `"like"` đứng riêng lẻ**, mà là một trạng thái ẩn mô tả cảm xúc tiêu cực: *"sự phủ định của lòng yêu thích"* (dislike / negative attitude).

#### 3. So sánh với Average Word2Vec:
- **Trong Average Word2Vec:**
  $$\vec{x}_{\text{doc}} = \frac{1}{6}(\vec{w}_{\text{I}} + \vec{w}_{\text{did}} + \vec{w}_{\text{not}} + \vec{w}_{\text{like}} + \vec{w}_{\text{this}} + \vec{w}_{\text{movie}})$$
  Phép cộng vector có tính chất giao hoán ($\vec{w}_{\text{not}} + \vec{w}_{\text{like}} = \vec{w}_{\text{like}} + \vec{w}_{\text{not}}$) và tính chất độc lập. Từ `"like"` vẫn đóng góp trọn vẹn vector dương tính vào tổng mà không bị từ `"not"` can thiệp.
- **Trong RNN:**
  Cơ chế hồi quy phá vỡ tính giao hoán. Thứ tự xuất hiện `"not"` đi liền trước `"like"` làm thay đổi trực tiếp môi trường tính toán của `"like"`, giúp mô hình nắm bắt chính xác quan hệ phủ định trong cú pháp học.

---

## Bài 3 — Thứ tự từ vựng (Sequence Order): "dog bites man" vs "man bites dog"

### 3.1. Đề bài
Cho hai chuỗi câu có cùng chính xác tập hợp từ vựng:
- $S_1 =$ `"dog bites man"`
- $S_2 =$ `"man bites dog"`

Yêu cầu:
1. Average Word2Vec có thể phân biệt hai câu này dựa trên thứ tự từ vựng hay không? Tại sao?
2. RNN phân biệt hai câu này như thế nào dựa trên cơ chế hoạt động bên trong?

---

### 3.2. Lời giải chi tiết

#### 1. Average Word2Vec hoàn toàn bất lực:
- Vector đại diện của câu $S_1$:
  $$\vec{x}(S_1) = \frac{1}{3}(\vec{w}_{\text{dog}} + \vec{w}_{\text{bites}} + \vec{w}_{\text{man}})$$
- Vector đại diện của câu $S_2$:
  $$\vec{x}(S_2) = \frac{1}{3}(\vec{w}_{\text{man}} + \vec{w}_{\text{bites}} + \vec{w}_{\text{dog}})$$
- Do phép cộng véc-tơ trên trường số thực có tính chất giao hoán:
  $$\vec{x}(S_1) = \vec{x}(S_2) \implies \text{Cosine Similarity}(\vec{x}(S_1), \vec{x}(S_2)) = 1.0$$
- Hai câu có nghĩa đối lập hoàn toàn về mặt quan hệ thực thể (ai là chủ thể gây hại, ai là nạn nhân chịu tác động), nhưng đối với Average Word2Vec, chúng là một đối tượng duy nhất.

#### 2. Cơ chế phân biệt của RNN dựa trên chuỗi trạng thái ẩn:
RNN xử lý câu như một quá trình Markov phi tuyến tính phụ thuộc thời gian:

* **Đối với câu $S_1$ ("dog bites man"):**
  1. $t=1$: Đọc `"dog"` $\to h_1 = \tanh(W_{xh} x_{\text{"dog"}} + b_h)$. Trạng thái ẩn $h_1$ mã hóa `"dog"` với tư cách là **chủ ngữ (Agent / Subject)** bắt đầu câu.
  2. $t=2$: Đọc `"bites"` $\to h_2 = \tanh(W_{xh} x_{\text{"bites"}} + W_{hh} h_1 + b_h)$. Trạng thái $h_2$ kết hợp hành động cắn với chủ thể `"dog"` (chó đang cắn).
  3. $t=3$: Đọc `"man"` $\to h_3^{(S_1)} = \tanh(W_{xh} x_{\text{"man"}} + W_{hh} h_2 + b_h)$. Trạng thái cuối cùng $h_3^{(S_1)}$ xác định `"man"` là **tân ngữ chịu tác động (Patient / Object)**.

* **Đối với câu $S_2$ ("man bites dog"):**
  1. $t=1$: Đọc `"man"` $\to h_1' = \tanh(W_{xh} x_{\text{"man"}} + b_h)$. Trạng thái $h_1'$ mã hóa `"man"` làm **chủ ngữ**.
  2. $t=2$: Đọc `"bites"` $\to h_2' = \tanh(W_{xh} x_{\text{"bites"}} + W_{hh} h_1' + b_h)$. Vì $h_1' \neq h_1$ nên $h_2' \neq h_2$ (người đang thực hiện hành động cắn).
  3. $t=3$: Đọc `"dog"` $\to h_3^{(S_2)} = \tanh(W_{xh} x_{\text{"dog"}} + W_{hh} h_2' + b_h)$. Trạng thái cuối cùng $h_3^{(S_2)}$ xác định `"dog"` là **tân ngữ**.

* **Kết luận:**
  Do $W_{xh} x_{\text{"man"}} \neq W_{xh} x_{\text{"dog"}}$ và $W_{hh} h_2 \neq W_{hh} h_2'$, ta có:
  $$h_3^{(S_1)} \neq h_3^{(S_2)}$$
  RNN phân biệt rõ ràng hai câu nhờ tính bất đối xứng theo thời gian của phương trình trạng thái đệ quy.

---

## Bài 4 — Hiện tượng triệt tiêu Gradient (Vanishing Gradient) trong RNN

### 4.1. Đề bài
Trong giải thuật Lan truyền ngược qua thời gian (Backpropagation Through Time — BPTT), gradient của hàm mất mát $L$ tại bước cuối $T$ đối với trạng thái ẩn tại bước $t$ ($t \ll T$) có dạng tích chuỗi:
$$\frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial h_T} \prod_{k=t+1}^T \frac{\partial h_k}{\partial h_{k-1}}$$

Yêu cầu:
1. Phân tích thành phần ma trận Jacobian $\frac{\partial h_k}{\partial h_{k-1}}$.
2. Giải thích vì sao khi độ dài chuỗi $T - t$ lớn, gradient có xu hướng triệt tiêu về $0$ ($\to 0$) hoặc bùng nổ ra vô cùng ($\to \infty$).
3. Nêu hệ quả đối với khả năng học các phụ thuộc xa (Long-range dependencies) của RNN.

---

### 4.2. Lời giải chi tiết

#### 1. Phân tích ma trận Jacobian:
Từ phương trình trạng thái $h_k = \tanh(W_{xh} x_k + W_{hh} h_{k-1} + b_h) = \tanh(a_k)$, đạo hàm riêng từng thành phần được tính theo quy tắc chuỗi:
$$\frac{\partial h_k}{\partial h_{k-1}} = \text{diag}(1 - h_k^2) W_{hh}^T$$
Trong đó:
- $1 - h_k^2$ là đạo hàm bậc nhất của hàm $\tanh(a_k)$. Vì $\tanh(z) \in (-1, 1)$ nên đạo hàm $1 - \tanh^2(z) \in (0, 1]$, đạt giá trị cực đại bằng $1$ tại $z=0$ và tiệm cận $0$ rất nhanh khi $|z|$ lớn (vùng bão hòa).
- $W_{hh}^T$ là ma trận trọng số hồi quy chuyển tiếp trạng thái.

#### 2. Cơ chế toán học của Vanishing và Exploding Gradient:
Tích chuỗi đạo hàm lan truyền qua $m = T - t$ bước:
$$\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^T \text{diag}(1 - h_k^2) W_{hh}^T$$
Áp dụng phân tích phổ (spectral decomposition), giả sử trị riêng lớn nhất của $W_{hh}$ là $\lambda_{\max}$:
$$\left\| \frac{\partial h_T}{\partial h_t} \right\| \le \prod_{k=t+1}^T \left\| \text{diag}(1 - h_k^2) \right\| \cdot \|W_{hh}^T\| \le (\gamma \cdot \lambda_{\max})^{T-t}$$
với $\gamma \le 1$.

- **Trường hợp $\gamma \cdot \lambda_{\max} < 1$ (Vanishing Gradient):**
  Lũy thừa bậc cao $m = T - t$ của một số nhỏ hơn 1 sẽ suy giảm về 0 theo hàm mũ:
  $$\lim_{(T-t) \to \infty} (\gamma \cdot \lambda_{\max})^{T-t} = 0 \implies \frac{\partial L}{\partial h_t} \approx 0$$
- **Trường hợp $\gamma \cdot \lambda_{\max} > 1$ (Exploding Gradient):**
  Lũy thừa bậc cao bùng nổ ra vô cực, làm trọng số nhận giá trị `NaN` hoặc dao động mất kiểm soát.

#### 3. Hệ quả đối với khả năng học của mô hình:
- Trọng số $W_{xh}$ và $W_{hh}$ được cập nhật bằng tổng gradient từ tất cả các bước thời gian:
  $$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^T \frac{\partial L}{\partial h_t} \frac{\partial h_t}{\partial W_{hh}}$$
- Khi $\frac{\partial L}{\partial h_t} \to 0$ với các bước $t$ ở xa, **mô hình hoàn toàn không nhận được tín hiệu phản hồi lỗi để cập nhật trọng số cho các từ ở đầu câu**.
- RNN chỉ học được các mối liên hệ cục bộ trong khoảng $5 - 10$ từ gần nhất, bị "mất trí nhớ tầm xa" và thất bại trước các câu văn dài có từ khóa quan trọng nằm cách xa nhau.

---

## Bài 5 — Tính toán trạng thái các cổng trong LSTM (LSTM Gates Computation)

### 5.1. Đề bài
Xét một bước thời gian $t$ trong kiến trúc Long Short-Term Memory (LSTM) với các giá trị cổng và trạng thái ứng viên đã được tính toán:
- Giá trị cổng quên (Forget gate): $f_t = 0.9$
- Giá trị cổng vào (Input gate): $i_t = 0.2$
- Giá trị trạng thái ứng viên (Candidate state): $\tilde{c}_t = 0.5$
- Trạng thái ô nhớ bước trước (Previous cell state): $c_{t-1} = 0.8$

Yêu cầu:
1. Áp dụng phương trình trạng thái ô nhớ để tính $c_t$:
   $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
2. Phân tích ý nghĩa điều khiển luồng thông tin khi cổng quên $f_t \to 1$.
3. Phân tích ý nghĩa điều khiển luồng thông tin khi cổng quên $f_t \to 0$.

---

### 5.2. Lời giải chi tiết

#### 1. Tính toán trạng thái ô nhớ mới $c_t$:
Thay các giá trị số vào phương trình:
$$c_t = (0.9 \times 0.8) + (0.2 \times 0.5)$$
$$c_t = 0.72 + 0.10 = 0.82$$

#### 2. Khi cổng quên $f_t \to 1$ (Gần bằng 1, ví dụ $f_t = 0.9$):
- **Cơ chế:** Thành phần $f_t \odot c_{t-1} \approx c_{t-1}$, nghĩa là ô nhớ giữ lại gần như nguyên vẹn $100\%$ tri thức và ký ức được tích lũy từ quá khứ xa xôi.
- **Ý nghĩa khắc phục Vanishing Gradient:** Đạo hàm của $c_t$ theo $c_{t-1}$ chứa số hạng cộng tính trực tiếp:
  $$\frac{\partial c_t}{\partial c_{t-1}} = f_t + \dots$$
  Khi $f_t \approx 1$, đạo hàm lan truyền qua ô nhớ là $\prod f_k \approx 1^m = 1$. Điều này tạo ra một **"đường cao tốc gradient" (Gradient Highway)** không bị suy giảm theo hàm mũ, cho phép mô hình LSTM truyền tín hiệu học tập xuyên suốt hàng trăm bước thời gian mà không bị triệt tiêu gradient như Simple RNN.

#### 3. Khi cổng quên $f_t \to 0$ (Gần bằng 0):
- **Cơ chế:** Thành phần $f_t \odot c_{t-1} \approx 0$. Mô hình chủ động xóa sạch toàn bộ thông tin lịch sử lưu trong cell state $c_{t-1}$.
- **Ý nghĩa ngôn ngữ học:** Hiện tượng này diễn ra khi mô hình nhận diện được ranh giới ngữ nghĩa rõ rệt (kết thúc một câu, chuyển sang một đoạn văn mới, hoặc chuyển đổi hoàn toàn chủ đề bàn luận). Việc xóa ký ức cũ giúp ô nhớ được làm mới hoàn toàn, sẵn sàng tiếp nhận thông tin mới từ vế sau ($i_t \odot \tilde{c}_t$) mà không bị nhiễu bởi ngữ cảnh không liên quan trước đó.
