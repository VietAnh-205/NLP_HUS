# LAB 07 — Báo cáo Tính toán Giải tích: Multi-Head Attention & Transformer Block

---

## 1. Bài tập 1: Tính toán một Attention Head (Mục 6)

### Đề bài
Cho:
- Đầu vào: $X \in \mathbb{R}^{3 \times 4}$ (chuỗi 3 tokens, số chiều embedding 4):
  $$X = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \end{bmatrix}$$
- Chiều không gian con: $d_k = 2 \implies \sqrt{d_k} = \sqrt{2} \approx 1.4142$.
- Ma trận chiếu:
  $$W^Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad W^K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 0 & 0 \end{bmatrix}, \quad W^V = \begin{bmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}$$

---

### Bước 1: Chiếu tuyến tính tạo $Q, K, V$
$$Q = X W^Q = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 2 & 0 \\ 0 & 2 \\ 1 & 1 \end{bmatrix}$$

$$K = X W^K = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 2 & 1 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}$$

$$V = X W^V = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 2 & 1 \\ 0 & 2 \\ 1 & 2 \end{bmatrix}$$

---

### Bước 2: Điểm tích vô hướng $Q K^T$ và Scaled Scores $S$
$$Q K^T = \begin{bmatrix} 2 & 0 \\ 0 & 2 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 2 & 0 & 1 \\ 1 & 1 & 1 \end{bmatrix} = \begin{bmatrix} 4 & 0 & 2 \\ 2 & 2 & 2 \\ 3 & 1 & 2 \end{bmatrix}$$

Chia tỉ lệ $\sqrt{2} \approx 1.4142$:
$$S = \frac{Q K^T}{\sqrt{2}} = \begin{bmatrix} 2.8284 & 0.0000 & 1.4142 \\ 1.4142 & 1.4142 & 1.4142 \\ 2.1213 & 0.7071 & 1.4142 \end{bmatrix}$$

---

### Bước 3: Trọng số Attention $A = \text{softmax}(S)$
Áp dụng Softmax theo từng hàng:
- **Hàng 1:** $\exp([2.8284, 0, 1.4142]) = [16.9184, 1.0000, 4.1132]$, Tổng $= 22.0316$.
  $$A_{1,:} = [0.7679, 0.0454, 0.1867]$$
- **Hàng 2:** Scores bằng nhau $[1.4142, 1.4142, 1.4142]$.
  $$A_{2,:} = [0.3333, 0.3333, 0.3333]$$
- **Hàng 3:** $\exp([2.1213, 0.7071, 1.4142]) = [8.3420, 2.0281, 4.1132]$, Tổng $= 14.4833$.
  $$A_{3,:} = [0.5760, 0.1400, 0.2840]$$

$$A = \begin{bmatrix} 0.7679 & 0.0454 & 0.1867 \\ 0.3333 & 0.3333 & 0.3333 \\ 0.5760 & 0.1400 & 0.2840 \end{bmatrix}$$

---

### Bước 4: Đầu ra Attention Head $Z = A V$
$$Z = \begin{bmatrix} 0.7679 & 0.0454 & 0.1867 \\ 0.3333 & 0.3333 & 0.3333 \\ 0.5760 & 0.1400 & 0.2840 \end{bmatrix} \begin{bmatrix} 2 & 1 \\ 0 & 2 \\ 1 & 2 \end{bmatrix} = \begin{bmatrix} 1.7225 & 1.2321 \\ 1.0000 & 1.6667 \\ 1.4360 & 1.4240 \end{bmatrix}$$

---

## 2. Bài tập 2: Cơ chế Multi-Head & Ghép nối (Mục 7)

### Đề bài
Cho $h = 2$ attention heads, mỗi head có số chiều $d_v = 2$:
$$head_1 \in \mathbb{R}^{T \times 2}, \quad head_2 \in \mathbb{R}^{T \times 2}$$
Ghép nối (Concatenation):
$$\text{Concat}(head_1, head_2) \in \mathbb{R}^{T \times 4}$$

### Trả lời câu hỏi: Vì sao cần concatenation trước khi áp dụng $W^O$?
- **Bảo toàn thông tin đa không gian:** Concatenation ghép các vector biểu diễn độc lập của từng head cạnh nhau mà không làm triệt tiêu hay ghi đè các đặc trưng ngữ nghĩa riêng biệt.
- **Tạo đầu vào cho ma trận chiếu hợp nhất:** Ma trận $W^O \in \mathbb{R}^{(h \cdot d_v) \times d_{model}} = \mathbb{R}^{4 \times 4}$ sẽ học cách trộn trọng số tuyến tính từ cả 2 heads, đưa biểu diễn về đúng kích thước chuẩn $d_{model}$ để đi tiếp vào các tầng sau.

---

## 3. Bài tập 3: Tính toán Positional Encoding (Mục 9, 10)

### Công thức
$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$
Với $d = 4$:
- $i = 0 \implies 2i = 0 \implies \omega_0 = \frac{1}{10000^{0/4}} = 1.0$
- $i = 1 \implies 2i = 2 \implies \omega_1 = \frac{1}{10000^{2/4}} = \frac{1}{100} = 0.01$

---

### Giá trị tại từng vị trí $pos \in \{0, 1, 2\}$
1. **Tại $pos = 0$:**
   - $PE(0, 0) = \sin(0 \cdot 1) = 0.0000$
   - $PE(0, 1) = \cos(0 \cdot 1) = 1.0000$
   - $PE(0, 2) = \sin(0 \cdot 0.01) = 0.0000$
   - $PE(0, 3) = \cos(0 \cdot 0.01) = 1.0000$
   $$\mathbf{PE}(0) = [0.0000, 1.0000, 0.0000, 1.0000]$$

2. **Tại $pos = 1$:**
   - $PE(1, 0) = \sin(1 \cdot 1) = 0.8415$
   - $PE(1, 1) = \cos(1 \cdot 1) = 0.5403$
   - $PE(1, 2) = \sin(1 \cdot 0.01) = 0.0100$
   - $PE(1, 3) = \cos(1 \cdot 0.01) = 0.99995 \approx 1.0000$
   $$\mathbf{PE}(1) = [0.8415, 0.5403, 0.0100, 1.0000]$$

3. **Tại $pos = 2$:**
   - $PE(2, 0) = \sin(2 \cdot 1) = 0.9093$
   - $PE(2, 1) = \cos(2 \cdot 1) = -0.4161$
   - $PE(2, 2) = \sin(2 \cdot 0.01) = 0.0200$
   - $PE(2, 3) = \cos(2 \cdot 0.01) = 0.9998$
   $$\mathbf{PE}(2) = [0.9093, -0.4161, 0.0200, 0.9998]$$

**Nhận xét:** Mỗi vị trí $pos$ sở hữu một vector vị trí duy nhất và trực giao tương đối, cho phép mạng nơ-ron phân biệt được trật tự chuỗi.

---

## 4. Bài tập 4: Tính toán Feed-Forward Network (FFN) (Mục 13, 14)

### Đề bài
Cho:
- Đầu vào: $x = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$
- Tham số lớp 1: $W_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad b_1 = \begin{bmatrix} 0 \\ -1 \end{bmatrix}$
- Tham số lớp 2: $W_2 = \begin{bmatrix} 1 & 2 \\ 0 & 1 \end{bmatrix}, \quad b_2 = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$

---

### Bước 1: Trạng thái ẩn $h = \text{ReLU}(W_1 x + b_1)$
$$W_1 x + b_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 0 \\ -1 \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 0 \\ -1 \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$
$$h = \text{ReLU}\left(\begin{bmatrix} 1 \\ 1 \end{bmatrix}\right) = \begin{bmatrix} \max(0, 1) \\ \max(0, 1) \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

---

### Bước 2: Đầu ra $y = W_2 h + b_2$
$$y = \begin{bmatrix} 1 & 2 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} + \begin{bmatrix} 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 1(1) + 2(1) \\ 0(1) + 1(1) \end{bmatrix} = \begin{bmatrix} 3 \\ 1 \end{bmatrix}$$

---

## 5. Bài tập 5: Phân tích Residual Connection (Mục 15)

### Đề bài
$$y = x + F(x)$$

### Trả lời câu hỏi: Nếu chỉ dùng $F(x)$, điều gì khác với $x + F(x)$?
- **Khác biệt luồng thông tin:** $F(x)$ buộc khối biến đổi phải học toàn bộ biểu diễn từ đầu, dễ làm thất thoát đặc trưng gốc. $x + F(x)$ cho phép khối chỉ cần học phần biến thiên dư (residual mapping $\Delta x$).
- **Khác biệt gradient:** 
  $$\frac{\partial y}{\partial x} = I + \frac{\partial F(x)}{\partial x}$$
  Số hạng đồng nhất $I$ đóng vai trò đường dẫn tắt (gradient highway), giữ đạo hàm luôn $\ge 1$, ngăn chặn triệt để hiện tượng tiêu biến gradient khi xếp chồng sâu nhiều Transformer blocks.

---

## 6. Bài tập 6: Kiểm tra Kích thước Tensor (Shape Check) (Mục 18)

Cấu hình: Batch size $B = 8$, Sequence length $T = 100$, Model dimension $d_{model} = 128$, Number of heads $h = 4 \implies d_k = d_v = \frac{128}{4} = 32$, $d_{ff} = 512$.

| Tầng / Bước tính toán | Ký hiệu toán học | Kích thước Tensor (Tensor Shape) |
| :--- | :---: | :---: |
| **Input Tensor** | $X$ | `(8, 100, 128)` |
| **Linear Projections ($Q, K, V$)** | $Q, K, V$ | `(8, 4, 100, 32)` |
| **Attention Scores** | $Q K^T / \sqrt{d_k}$ | `(8, 4, 100, 100)` |
| **Attention Weights** | $A = \text{softmax}(S)$ | `(8, 4, 100, 100)` |
| **Head Context Output** | $A V$ | `(8, 4, 100, 32)` |
| **Concatenated Heads** | $\text{Concat}(head_1, \dots, head_h)$ | `(8, 100, 128)` |
| **Multi-Head Linear Output** | $MHA(X)$ | `(8, 100, 128)` |
| **Add & Norm 1** | $\text{LayerNorm}(X + MHA(X))$ | `(8, 100, 128)` |
| **FFN Hidden Layer** | $\text{ReLU}(X W_1 + b_1)$ | `(8, 100, 512)` |
| **FFN Output Layer** | $FFN(X)$ | `(8, 100, 128)` |
| **Add & Norm 2 (Block Output)** | $\text{LayerNorm}(X + FFN(X))$ | `(8, 100, 128)` |

---

## 7. Bài tập 7: Phân tích Chi phí Chuỗi $T^2$ (Mục 22)

| Độ dài chuỗi ($T$) | Kích thước ma trận Attention ($T \times T$) | Số phần tử ($T^2$) | Bộ nhớ 1 ma trận (float32) | FLOPs cho $Q K^T$ ($2 B h T^2 d_k$) |
| :---: | :---: | :---: | :---: | :---: |
| **32** | $32 \times 32$ | $1,024$ | $4.0 \text{ KB}$ | $2.1 \times 10^6$ |
| **64** | $64 \times 64$ | $4,096$ | $16.0 \text{ KB}$ | $8.4 \times 10^6$ |
| **128** | $128 \times 128$ | $16,384$ | $64.0 \text{ KB}$ | $3.36 \times 10^7$ |
| **256** | $256 \times 256$ | $65,536$ | $256.0 \text{ KB}$ | $1.34 \times 10^8$ |

### Trả lời câu hỏi: Tại sao Transformer không thể đơn giản giải quyết mọi vấn đề bằng cách tăng sequence length?
1. **Nghẽn cổ chai bậc hai $\mathcal{O}(T^2)$:** Số lượng phần tử ma trận và phép tính tăng gấp 4 lần mỗi khi $T$ tăng gấp đôi.
2. **Tràn bộ nhớ GPU (Out-Of-Memory):** Ma trận Attention weights và activation cache của tất cả các lớp/đầu trong quá trình Backpropagation nhanh chóng làm tràn VRAM nếu chuỗi quá dài.

