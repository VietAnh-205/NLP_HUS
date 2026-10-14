# LAB 04 — REFLECTION: SUY NGẪM LÝ THUYẾT VÀ TIẾN HÓA BIỂU DIỄN



## 1. Bảng so sánh toàn diện hai trường phái biểu diễn văn bản

| Đặc tính so sánh | TF-IDF Representation (Sparse Lexical) | Average Word2Vec (Dense Semantic) |
|:---|:---|:---|
| **Cơ sở lý thuyết** | Giả thuyết thống kê tần suất từ vựng (Salton & Buckley, 1988). | Giả thuyết phân bố ngôn ngữ (Harris, 1954; Mikolov et al., 2013). |
| **Không gian vector** | Không gian rời rạc, thưa (Sparse), số chiều rất lớn ($d = |V| \in [5,000, 50,000]$). | Không gian liên tục, đặc (Dense), số chiều thấp ($d \in [50, 300]$). |
| **Bản chất trích xuất** | **Lexical:** Mỗi từ là một chiều độc lập (Orthogonal), giữ nguyên danh tính bề mặt của từ. | **Semantic:** Các từ nằm chung trong không gian liên tục, từ có ngữ cảnh giống nhau sẽ ở gần nhau. |
| **Cơ chế đánh trọng số** | $\text{TF} \times \text{IDF}$: Tự động triệt tiêu từ phổ thông và khuếch đại từ khóa đặc trưng. | Mặc định bình đẳng giữa các từ; từ phổ thông và từ then chốt được cộng dồn ngang nhau. |
| **Quan hệ từ trái nghĩa** | `good` và `bad` là 2 vector vuông góc ($\text{Cosine} = 0$), cực kỳ dễ phân tách tuyến tính. | `good` và `bad` có $\text{Cosine} > 0.7$ vì cùng xuất hiện sau *"the movie is very..."*. |
| **Chi phí tính toán** | Cực kỳ nhanh, chỉ cần đếm và tính nghịch đảo tần suất tài liệu. | Đòi hỏi huấn luyện mạng nơ-ron hoặc phân tích ma trận SVD trên tập ngữ liệu. |
| **Khả năng tổng quát hóa** | Kém khi gặp từ đồng nghĩa chưa có trong tập train (Lexical Mismatch). | Tốt hơn khi gặp từ đồng nghĩa nhờ không gian nhúng ngữ nghĩa liên tục. |
| **Hiệu năng trên Sentiment** | **Cao hơn ($82.00\% - 84.67\%$)** do bảo toàn được từ khóa cảm xúc đặc thù. | **Thấp hơn ($75.00\%$)** do hiện tượng hòa tan thông tin (Information Dilution). |

---

## 2. Trả lời câu hỏi trọng tâm: Tại sao hai representation tạo ra kết quả khác nhau?

Hai biểu diễn dẫn đến hiệu năng phân loại khác biệt sâu sắc bắt nguồn từ **sự tương thích giữa bản chất bài toán và cấu trúc không gian đặc trưng**:

1. **Bản chất của bài toán Phân loại Cảm xúc (Sentiment Analysis):**
   - Trong một bài phê bình dài 700 từ, có đến $95\%$ dung lượng văn bản là mô tả bối cảnh, nhân vật, tóm tắt cốt truyện (mang tính trung tính). Cảm xúc thực sự của bài viết thường chỉ được gói gọn trong một vài tính từ hoặc trạng từ then chốt (ví dụ: `masterpiece`, `boring`, `unwatchable`).
2. **Cơ chế của TF-IDF:**
   - TF-IDF hoạt động như một **bộ lọc chọn lọc cục bộ**: Các từ xuất hiện ở mọi nơi sẽ có IDF $\approx 0$, trong khi các từ cảm xúc đặc thù sẽ có trọng số TF-IDF nổi bật. Bộ phân loại Logistic Regression dễ dàng học được các trọng số $w_i > 0$ hoặc $w_i < 0$ trên từng chiều từ vựng trực giao đó.
3. **Sự thất bại của phép trung bình cộng trong Word2Vec:**
   - Phép tính $\vec{x}_{\text{doc}} = \frac{1}{n} \sum_{i=1}^n \vec{w}_i$ đối xử cào bằng với mọi từ trong câu. Vector của một từ cảm xúc mạnh mẽ như `brilliant` bị cộng dồn và chia trung bình với hàng trăm vector danh từ trung tính (`movie`, `actor`, `screen`, `house`). Kết quả là vector của toàn bài bị kéo mạnh về "tâm khối" (centroid) của toàn bộ từ vựng, làm mất hoàn toàn độ sắc nhọn của tín hiệu cảm xúc.

---

## 3. Thí nghiệm *"this movie is not good"* và Giới hạn của Bag-of-Words

Thực nghiệm ở Mục 17 đã chỉ ra:
$$\text{Cosine Similarity}(\vec{x}_{\text{"this movie is good"}}, \vec{x}_{\text{"this movie is not good"}}) = \mathbf{0.9898}$$
Hai câu mang cảm xúc đối lập $180^\circ$ nhưng lại có khoảng cách hình học gần như trùng khít trong không gian Average Word2Vec!

### Nguyên nhân cốt lõi:
1. **Sự biến mất của thứ tự từ (Loss of Word Order):** Cả mô hình n-gram unigram và Average Embeddings đều dựa trên giả định túi từ (Bag-of-Words), xem văn bản là một tập hợp các từ không có liên kết vị trí.
2. **Tác động cộng dồn đơn thuần:** Từ phủ định `not` chỉ đóng góp thêm một vector $\vec{w}_{\text{not}}$ vào tổng véc-tơ của văn bản, thay vì đóng vai trò như một toán tử logic đảo ngược cực tính của từ đi liền sau nó (`not` $\times$ `good` $\to$ `bad`).
3. **Bước chuyển sang Deep Learning:**
   Đây chính là lý do nền tảng chứng minh các mô hình tĩnh không thể giải quyết được các cấu trúc ngôn ngữ phức tạp, đòi hỏi phải tiến hóa sang:
   - **Mạng nơ-ron hồi quy (RNN / LSTM):** Lưu vết trạng thái ẩn $h_t = f(h_{t-1}, x_t)$ qua từng bước thời gian.
   - **Cơ chế Attention và Transformer (BERT, GPT):** Sử dụng Self-Attention để từ `good` tự động "chú ý" (attend) vào từ `not` đứng ngay trước nó, từ đó sinh ra biểu diễn ngữ cảnh động (Contextualized Representation) hoàn toàn phân biệt.

---

## 4. Trả lời chi tiết 6 câu hỏi kiểm tra vấn đáp cá nhân (Individual Learning Check)

### Câu 1: Tại sao Accuracy 95% chưa chắc đã là một mô hình tốt?
* **Trả lời:**
  Accuracy chỉ đo lường tỷ lệ dự đoán đúng tổng thể trên toàn bộ tập dữ liệu: $\text{Accuracy} = \frac{\text{TP} + \text{TN}}{N}$. Khi tập dữ liệu bị mất cân bằng lớp trầm trọng (Class Imbalance, ví dụ $95\%$ lớp Negative và $5\%$ lớp Positive), một mô hình tầm thường chỉ cần đoán mọi mẫu là Negative cũng sẽ đạt Accuracy $95\%$. Tuy nhiên, mô hình này hoàn toàn vô dụng vì có $\text{Recall}_{\text{pos}} = 0\%$, bỏ sót toàn bộ các trường hợp Positive (như bỏ sót toàn bộ bệnh nhân ung thư hoặc giao dịch gian lận). Để đánh giá đúng, bắt buộc phải xem xét **Confusion Matrix, Balanced Accuracy, Precision, Recall và Macro $F_1$-score**.

### Câu 2: TF-IDF khác Average Word2Vec ở những điểm cốt lõi nào?
* **Trả lời:**
  TF-IDF tạo ra không gian vector **rời rạc, thưa (Sparse) và số chiều rất lớn ($d = |V|$)**, trong đó mỗi từ là một chiều độc lập (Orthogonal) và giá trị mỗi chiều phản ánh tầm quan trọng thống kê của từ khóa thông qua tần suất xuất hiện. Ngược lại, Average Word2Vec tạo ra không gian vector **liên tục, đặc (Dense) và số chiều thấp ($d \in [50, 300]$)**, trong đó các từ chia sẻ chung các chiều tiềm ẩn ngữ nghĩa được học từ phân bố ngữ cảnh. TF-IDF bảo toàn sự phân biệt của từng từ vựng cụ thể, còn Average Word2Vec nén ngữ nghĩa nhưng dễ làm loãng các từ khóa quan trọng khi lấy trung bình toàn văn bản.

### Câu 3: Tại sao cụm từ *"not good"* có thể gây lỗi nghiêm trọng cho các mô hình truyền thống?
* **Trả lời:**
  Vì các mô hình truyền thống (Unigram TF-IDF hoặc Average Embeddings) hoạt động theo cơ chế Bag-of-Words. Mô hình xem `"not"` và `"good"` là hai thực thể tách rời được cộng điểm độc lập. Từ `"good"` mang trọng số tích cực mạnh, trong khi `"not"` là một hư từ phổ thông thường có trọng số yếu hoặc thậm chí bị xóa trong quá trình loại bỏ Stopwords. Do đó, mô hình vẫn nhìn thấy tín hiệu tích cực áp đảo của `"good"` và phân loại sai câu thành Positive, hoàn toàn không nhận thức được quan hệ cú pháp phủ định giữa `"not"` và `"good"`.

### Câu 4: Hiện tượng Data Leakage (Rò rỉ dữ liệu) xảy ra ở đâu nếu fit TF-IDF trước khi chia Train/Test?
* **Trả lời:**
  Data Leakage xảy ra ở hai điểm then chốt:
  1. **Rò rỉ từ vựng (Vocabulary):** Tập Test chứa các từ ngữ chỉ có trong tương lai. Fit trước khi split khiến bộ trích xuất đưa các từ của Test vào từ điển huấn luyện, vi phạm nguyên tắc mô hình không được biết trước tương lai.
  2. **Rò rỉ tần số nghịch đảo (IDF):** Trọng số $\text{IDF}(t) = \log\frac{N}{\text{DF}(t)}$ phụ thuộc vào số lượng tài liệu chứa từ $t$. Nếu tính DF trên cả tập Test, độ quan trọng của từ trong tập Train đã bị can thiệp bởi phân phối của tập Test. Kết quả là điểm số đánh giá trên tập Test trở nên cao một cách giả tạo và không phản ánh đúng năng lực khi triển khai thực tế.

### Câu 5: Precision và Recall khác nhau như thế nào và khi nào ưu tiên chỉ số nào?
* **Trả lời:**
  - **Precision** ($\frac{\text{TP}}{\text{TP} + \text{FP}}$): Đo lường độ tin cậy của lời dự đoán dương tính — *"Trong các mẫu mô hình bảo là Positive, có bao nhiêu % đúng là Positive?"*. Ưu tiên Precision cao khi **chi phí của False Positive rất đắt** (ví dụ: Lọc thư rác Spam — không được phép xóa nhầm email quan trọng của người dùng; hay Gợi ý video — video gợi ý phải thực sự hấp dẫn).
  - **Recall** ($\frac{\text{TP}}{\text{TP} + \text{FN}}$): Đo lường độ bao phủ — *"Trong toàn bộ các mẫu Positive ngoài đời thực, mô hình đã tóm được bao nhiêu %?"*. Ưu tiên Recall cao khi **chi phí của False Negative là thảm họa** (ví dụ: Chẩn đoán ung thư, Phát hiện gian lận tài chính, Cảnh báo khủng bố — thà bắt nhầm còn hơn bỏ sót).

### Câu 6: Tại sao việc thêm đặc trưng Bigram có thể cải thiện phân loại cảm xúc?
* **Trả lời:**
  Thêm Bigram (`ngram_range=(1, 2)`) cho phép mô hình chụp lại các cặp từ liền kề có liên kết ngữ nghĩa chặt chẽ. Cụ thể, các cụm từ đảo ngược cảm xúc như `"not good"`, `"never watch"`, `"hardly recommend"` hoặc các cụm từ nhấn mạnh cảm xúc như `"very bad"`, `"must see"`, `"waste of time"` sẽ được xem như một đặc trưng độc lập với trọng số riêng biệt. Khi đó, cụm `"not good"` sẽ có trọng số âm trực tiếp, vô hiệu hóa tín hiệu dương của từ đơn `"good"`, từ đó giải quyết triệt để lỗi phân loại do hiện tượng phủ định cục bộ gây ra.

---

## 5. Tuyên bố sử dụng Trí tuệ Nhân tạo (AI Assistance Statement)

Theo quy định tại **Mục 24. AI Policy** của đề cương Lab:
- **Công cụ sử dụng (Tool):** Antigravity IDE (Gemini 3.8 Flash Engine).
- **Mục đích hỗ trợ (Purpose):** Hỗ trợ tối ưu hóa cấu trúc code trong `preprocessing.py`, tự động hóa sinh script chạy thực nghiệm, vẽ biểu đồ ma trận nhầm lẫn (`confusion_matrix.png`) và định dạng tài liệu Markdown chuẩn công thức toán học LaTeX.
- **Nội dung sinh tự động (Generated):** Khung sườn notebook `classification.ipynb`, hàm xử lý ma trận thưa TruncatedSVD trong `preprocessing.py`.
- **Nội dung do sinh viên chỉnh sửa & tối ưu (Modified):** Tinh chỉnh ngưỡng siêu tham số $V = 5000$, điều chỉnh tỷ lệ phân chia Train/Val/Test chống rò rỉ dữ liệu, xây dựng cấu trúc bảng số liệu so sánh.
- **Nội dung sinh viên tự thực hiện và kiểm chứng 100% (Verified & Authored):** Toàn bộ các bài tập tính toán tay (`calculations.md`), thiết lập 5 giả thuyết dự đoán (`prediction.md`), phân tích bản chất định tính 8 nhóm lỗi (`error_analysis.md`) và trả lời các câu hỏi suy ngẫm lý thuyết chuyên sâu (`reflection.md`).
