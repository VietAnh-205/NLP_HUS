# LAB 09 — Bài tập Tính toán: BIO Tagging, Entity Boundary & Tensor Shapes

---

## 1. Calculation Exercise 1 — BIO Tagging (Mục 6)

**Câu đầu vào:** `Elon Musk founded SpaceX in California.`  
**Thực thể cho trước:** `Elon Musk` (PER), `SpaceX` (ORG), `California` (LOC).

| Token | BIO Label | Giải thích ngắn |
| :--- | :---: | :--- |
| `Elon` | `B-PER` | Token mở đầu thực thể PER |
| `Musk` | `I-PER` | Token tiếp nối bên trong thực thể PER |
| `founded` | `O` | Ngoài thực thể |
| `SpaceX` | `B-ORG` | Thực thể ORG đơn token |
| `in` | `O` | Ngoài thực thể |
| `California` | `B-LOC` | Thực thể LOC đơn token |
| `.` | `O` | Dấu câu (ngoài thực thể) |

---

## 2. Calculation Exercise 2 — Entity Boundary (Mục 7)

**Cụm đầu vào:** `University of California Berkeley` (1 thực thể ORG duy nhất).

| Token | BIO Label |
| :--- | :---: |
| `University` | `B-ORG` |
| `of` | `I-ORG` |
| `California` | `I-ORG` |
| `Berkeley` | `I-ORG` |

**Vì sao không thể gán tất cả token là `ORG` nếu muốn biểu diễn ranh giới entity?**
- Nếu không phân biệt `B-` (Beginning) và `I-` (Inside), hai thực thể cùng loại đứng liền nhau sẽ bị gộp nhầm thành một thực thể duy nhất (ví dụ: `Google Microsoft` gán `ORG ORG` không rõ là 1 hay 2 tổ chức).
- Thẻ `B-ORG` đánh dấu chính xác điểm khởi đầu của thực thể mới, đảm bảo tách biệt ranh giới (boundary) tuyệt đối.

---

## 3. Calculation Exercise 3 — Tensor Shapes (Mục 15)

**Tham số đầu vào:** $B = 8$, $T = 128$, $d = 768$, $C = 9$ (9 nhãn CoNLL-2003: `O`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`, `B-MISC`, `I-MISC`).

| Thành phần | Ký hiệu | Shape | Ghi chú tính toán |
| :--- | :---: | :---: | :--- |
| **Hidden states** | $H$ | `(8, 128, 768)` | Đầu ra của Transformer Encoder tại mọi vị trí token |
| **Linear weight & bias** | $W, b$ | `(768, 9)` và `(9,)` | Classification head áp dụng chung cho từng token |
| **Logits** | $Z = HW + b$ | `(8, 128, 9)` | Mỗi token có 1 vector logits $z_i \in \mathbb{R}^9$ |
| **Labels** | $Y$ | `(8, 128)` | Nhãn nguyên $y_{b,t} \in \{0,\dots,8\} \cup \{-100\}$ |

---

## 4. Tóm tắt công thức Ignore Index & Entity-Level Metrics (Mục 11, 19, 20)

- **Subword Alignment (Strategy A — First-subword):** Gán nhãn thực cho subword đầu tiên, gán `IGNORE` (`-100`) cho các subword còn lại và special tokens (`[CLS]`, `[SEP]`, `[PAD]`). Loss chỉ tính trên tập vị trí hợp lệ:
  $$\mathcal{L} = -\frac{1}{|\{i : y_i \neq -100\}|} \sum_{i : y_i \neq -100} \log P(y_i \mid x)$$
- **Token-level vs. Entity-level (Ví dụ `New York`):**
  - Gold: `[B-LOC, I-LOC]` $\to$ Entity `("New York", LOC)`
  - Pred: `[B-LOC, O]` $\to$ Entity `("New", LOC)`
  - **Token-level:** đúng 1/2 token ($50\%$).
  - **Entity-level (Exact Match):** $TP = 0, FP = 1, FN = 1 \implies P = 0, R = 0, F_1 = 0$.
