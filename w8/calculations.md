
---

## 1. Bài tập Tính toán 1: Hàm mất mát Masked Language Modeling (MLM Loss)

### Đề bài
Một token bị che trong câu: *"The cat [MASK] fish."*
Xác suất dự đoán phân phối từ vựng của mô hình:
- $P(\text{"cat"}) = 0.10$
- $P(\text{"dog"}) = 0.05$
- $P(\text{"eats"}) = 0.70$
- $P(\text{"fish"}) = 0.15$

Token nhãn đúng thực tế: $y = \text{"eats"}$.

---

### Bước 1: Tính Cross-Entropy Loss
Hàm mất mát Cross-Entropy (sử dụng logarit tự nhiên $\ln$ chuẩn mực trong NLP):
$$\mathcal{L} = -\ln(P(\text{"eats"})) = -\ln(0.70) \approx \mathbf{0.3567}$$
*(Nếu tính theo $\log_{10}$: $\mathcal{L} = -\log_{10}(0.70) \approx 0.1549$)*

---

### Bước 2: Trả lời 3 câu hỏi trọng tâm

1. **Nếu mô hình dự đoán $P(\text{"eats"}) = 0.95$:**
   $$\mathcal{L} = -\ln(0.95) \approx \mathbf{0.0513}$$
   - **Thay đổi:** Loss giảm mạnh từ $0.3567 \to 0.0513$ (giảm $85.62\%$). Mô hình được thưởng khi dự đoán đúng với độ tự tin cao.

2. **Nếu mô hình dự đoán $P(\text{"eats"}) = 0.01$:**
   $$\mathcal{L} = -\ln(0.01) \approx \mathbf{4.6052}$$
   - **Thay đổi:** Loss tăng vọt từ $0.3567 \to 4.6052$ (tăng $12.9$ lần). Mô hình bị trừng phạt rất nặng khi gán xác suất thấp cho token đúng.

3. **Vì sao Cross-Entropy phù hợp với bài toán MLM?**
   - **Nguyên lý MLE:** Tối thiểu hóa Cross-Entropy tương đương cực đại hóa hàm hợp lý (Maximum Likelihood) của dữ liệu văn bản.
   - **Gradient nhạy bén:** Đạo hàm của loss theo logit $z_i$ là $\frac{\partial \mathcal{L}}{\partial z_i} = P(y_i|x) - 1$. Khi xác suất đúng tiệm cận $0$, độ dốc đạt cực đại, tạo lực kéo gradient mạnh mẽ buộc mạng cập nhật trọng số.

---

## 2. Bài tập Tính toán 2: Phân tích Tokenization (BERT WordPiece)

### Đề bài
Câu văn bản đầu vào: *"The cats are playing."*
Tokenizer chuẩn: `bert-base-uncased` (WordPiece).

---

### Kết quả phân tích từng bước

1. **Token Sequence (Chuỗi Token):**
   `['[CLS]', 'the', 'cats', 'are', 'playing', '.', '[SEP]']`

2. **Token IDs (Mã số định danh):**
   `[101, 1996, 8870, 2024, 2652, 1012, 102]`

3. **Special Tokens (Token đặc biệt):**
   - `[CLS]` (ID = 101): Token phân loại ở đầu câu, đại diện cho vector toàn chuỗi.
   - `[SEP]` (ID = 102): Token phân cách kết thúc câu.

4. **Attention Mask (Mặt nạ chú ý):**
   `[1, 1, 1, 1, 1, 1, 1]`
   *(Tất cả đều bằng $1$ vì không có token đệm `[PAD]`)*

5. **So sánh số lượng:**
   - **Words (Từ tự nhiên):** $4$ từ (*"The"*, *"cats"*, *"are"*, *"playing"*).
   - **Tokens (Bao gồm Special Tokens & Dấu câu):** $7$ tokens.
   - **Subword Tokens:** 
     - Trong từ điển BERT uncased (kích thước $30,522$), các từ *"the"*, *"cats"*, *"are"*, *"playing"* đều nằm sẵn trong từ vựng nên không bị chia tách thêm subwords có tiền tố `##`.
     - Với từ ghép hiếm (OOV) như *"unbelievable"*, WordPiece sẽ tách thành: `['un', '##beli', '##eva', '##ble']`.

