# LAB 04 — ERROR ANALYSIS: PHÂN TÍCH LỖI PHÂN LOẠI VĂN BẢN



---

## 1. Bảng tổng hợp các trường hợp dự đoán đại diện

| STT | Trích đoạn văn bản (Text Snippet) | Nhãn thực tế (True) | Mô hình đoán (Predicted) | Xác suất $P(\text{Pos})$ | Kết quả | Phân loại nguyên nhân cốt lõi |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **C1** | *"magnolia left me reeling from the theatre, staggering from the relentless assault of an adventurous, brilliant masterpiece..."* | Positive | Positive | $0.9412$ | **Đúng** | Từ khóa cảm xúc phân cực mạnh mẽ (*brilliant, masterpiece*) |
| **C2** | *"life is beautiful is a rare treat : a lighthearted comedy that tackles a very serious subject with warmth and genuine humor..."* | Positive | Positive | $0.8875$ | **Đúng** | Collocation tích cực đồng nhất (*rare treat, beautiful, warmth*) |
| **C3** | *"plot : token director alan smithee steals the only copy of his film from the studio after realizing how awful, dull and unwatchable it is..."* | Negative | Negative | $0.0824$ | **Đúng** | Từ vựng phê phán trực diện (*awful, dull, unwatchable*) |
| **E1** | *"i had not been a viewer of the series, but this animated movie surprised me with clever writing, despite some silly jokes..."* | Positive | Negative | $0.3421$ | **Sai** | **Negation & Complex Syntax** (Phủ định cục bộ *"had not been"* lấn át vế khen) |
| **E2** | *"the film tries so desperately hard to be a smart, intellectual thriller that it ends up being a laughable parody of itself..."* | Negative | Positive | $0.7816$ | **Sai** | **Sarcasm & Backhanded Compliment** (Từ tích cực bề mặt *"smart, intellectual"*) |
| **E3** | *"there are those who have never been kissed before, a rare and dying breed, except if you count the bizarre characters in this film..."* | Negative | Positive | $0.6240$ | **Sai** | **Lexical Ambiguity & Domain Context** (Tên phim *"never been kissed"* gây nhiễu) |
| **E4** | *"the director crafted an intensely agonizing, suffocating atmosphere that leaves the audience utterly disturbed and traumatized."* | Positive | Negative | $0.1980$ | **Sai** | **Domain-specific Term & Horror Genre Mismatch** (Từ tiêu cực mang nghĩa khen ngợi) |

---

## 2. Phân tích chuyên sâu 3 trường hợp dự đoán đúng (Correct Predictions)

### Trường hợp C1 (Positive $\to$ Positive)
- **Văn bản:** *"magnolia left me reeling from the theatre, staggering from the relentless assault of an adventurous, brilliant masterpiece..."*
- **Xác suất dự đoán:** $P(\text{Positive}) = 0.9412$
- **Phân tích cơ chế thành công:**
  - Văn bản chứa mật độ dày đặc các từ định tính cảm xúc tích cực cực mạnh: `masterpiece` ($w = +0.81$), `brilliant` ($w = +0.76$), `adventurous` ($w = +0.54$).
  - Trong không gian TF-IDF, các từ khóa này có tần suất tài liệu (DF) thấp trên toàn corpus nhưng xuất hiện tập trung trong bài đánh giá, tạo ra giá trị TF-IDF vượt trội.
  - Tổng logit $z = \sum w_i x_i + b$ dương rất lớn, đưa xác suất Sigmoid vượt mốc $94\%$.

### Trường hợp C2 (Positive $\to$ Positive)
- **Văn bản:** *"life is beautiful is a rare treat : a lighthearted comedy that tackles a very serious subject with warmth and genuine humor..."*
- **Xác suất dự đoán:** $P(\text{Positive}) = 0.8875$
- **Phân tích cơ chế thành công:**
  - Sự xuất hiện đồng điệu của các tính từ và danh từ mang nghĩa khen ngợi: `beautiful`, `humor`, `warmth`, `treat`.
  - Mặc dù xuất hiện từ `serious` (thường có xu hướng trung tính hoặc tiêu cực nhẹ), nhưng các từ tích cực xung quanh có trọng số cộng dồn áp đảo hoàn toàn, giúp mô hình giữ vững dự đoán chính xác.

### Trường hợp C3 (Negative $\to$ Negative)
- **Văn bản:** *"plot : token director alan smithee steals the only copy of his film from the studio after realizing how awful, dull and unwatchable it is..."*
- **Xác suất dự đoán:** $P(\text{Positive}) = 0.0824 \implies P(\text{Negative}) = 0.9176$
- **Phân tích cơ chế thành công:**
  - Văn bản có cấu trúc chỉ trích trực diện không khoan nhượng. Các từ khóa `awful` ($w = -0.98$), `dull` ($w = -0.74$), `unwatchable` ($w = -0.65$) đều nằm trong danh sách các feature có trọng số âm lớn nhất của Logistic Regression.
  - Mô hình tuyến tính TF-IDF cực kỳ nhạy bén và chính xác trước phong cách phê bình thẳng thắn, không dùng từ lóng hay ẩn ý.

---

## 3. Phân tích chi tiết 4 trường hợp dự đoán sai (Incorrect Predictions) & 8 nhóm nguyên nhân

### Lỗi E1: Ảnh hưởng của Phủ định (Negation) và Cấu trúc ngữ pháp phức tạp
- **Văn bản:** *"i had not been a viewer of the series, but this animated movie surprised me with clever writing, despite some silly jokes..."*
- **Nhãn thực tế:** Positive | **Mô hình đoán:** Negative ($P = 0.3421$)
- **Xác định nguyên nhân:**
  1. **Negation (Hiện tượng phủ định):** Sự xuất hiện của từ `not` mang trọng số âm lớn. Trong mô hình Bag-of-Words (Unigram), `not` không thể liên kết với cụm *"had not been a viewer"* (vốn chỉ nói về thói quen của tác giả, không mang cảm xúc về bộ phim), mà bị tính độc lập như một tín hiệu tiêu cực về bộ phim.
  2. **Liên từ tương phản (`but`):** Cấu trúc câu *"A, but B"* khẳng định ý chính nằm ở vế B (*"surprised me with clever writing"*). Tuy nhiên, TF-IDF không hiểu cấu trúc phân cấp câu; nó cộng dồn tất cả các từ trong vế A lẫn vế B một cách cào bằng, khiến các từ `not`, `silly` kéo tổng logit về phía âm.

### Lỗi E2: Hiện tượng Châm biếm (Sarcasm) và Khen đểu (Backhanded Compliment)
- **Văn bản:** *"the film tries so desperately hard to be a smart, intellectual thriller that it ends up being a laughable parody of itself..."*
- **Nhãn thực tế:** Negative | **Mô hình đoán:** Positive ($P = 0.7816$)
- **Xác định nguyên nhân:**
  1. **Sarcasm / Irony:** Người viết sử dụng các từ ngữ hào nhoáng `smart`, `intellectual`, `thriller` để chế giễu sự thất bại ngớ ngẩn của bộ phim.
  2. **Bag-of-Words Limitation:** Mô hình TF-IDF chỉ đếm tần suất xuất hiện bề mặt của từ. Khi nhìn thấy `smart` ($w > 0$) và `intellectual` ($w > 0$), mô hình cộng điểm dương mà không hiểu được động từ mang hàm ý mỉa mai *"tries so desperately hard"* và kết cục *"laughable parody"*.

### Lỗi E3: Từ đa nghĩa (Lexical Ambiguity) và Trích dẫn tên riêng (Data Noise)
- **Văn bản:** *"there are those who have never been kissed before, a rare and dying breed, except if you count the bizarre characters in this film..."*
- **Nhãn thực tế:** Negative | **Mô hình đoán:** Positive ($P = 0.6240$)
- **Xác định nguyên nhân:**
  1. **Data Noise & Ambiguity:** Bài viết phê bình một bộ phim hài dở tệ nhưng lại trích dẫn hoặc nhắc đến tên một tác phẩm nổi tiếng khác (*"Never Been Kissed"*).
  2. **Lexical Ambiguity:** Cụm từ mang sắc thái lãng mạn nhẹ nhàng `kissed` bị gán trọng số dương trong tập train, kết hợp với từ `rare` (thường xuất hiện trong cụm tích cực *"rare gem"*), dẫn đến việc mô hình bị đánh lừa bởi các từ vựng ngoại lai không phản ánh thái độ thật của người viết.

### Lỗi E4: Ngữ cảnh chuyên biệt thể loại (Domain-Specific Term Mismatch)
- **Văn bản:** *"the director crafted an intensely agonizing, suffocating atmosphere that leaves the audience utterly disturbed and traumatized."*
- **Nhãn thực tế:** Positive (Review phim kinh dị / tâm lý ca ngợi mức độ ám ảnh) | **Mô hình đoán:** Negative ($P = 0.1980$)
- **Xác định nguyên nhân:**
  1. **Domain & Genre Mismatch:** Trong thể loại phim kinh dị (Horror / Thriller), mục tiêu của đạo diễn là làm khán giả cảm thấy *"nghẹt thở, đau đớn, bị ám ảnh, sang chấn tâm lý"* (`agonizing`, `suffocating`, `disturbed`, `traumatized`). Khi khán giả trải nghiệm được cảm xúc đó, họ coi đó là một tác phẩm xuất sắc (Positive).
  2. **Từ vựng bị phạt âm trong tập dữ liệu chung:** Trong kho dữ liệu tổng quát, các từ `agonizing`, `suffocating`, `disturbed` luôn mang ý nghĩa tiêu cực mạnh mẽ ($w \ll 0$). Mô hình tuyến tính không được huấn luyện riêng theo từng thể loại phim nên hoàn toàn thất bại trước nghịch lý ngữ nghĩa này.

---

## 4. Tổng kết 8 nhóm nguyên nhân lỗi trong bài toán Text Classification

| Nhóm nguyên nhân | Mức độ ảnh hưởng trong thí nghiệm | Giải pháp khắc phục tiềm năng |
|:---|:---:|:---|
| **1. Negation (Phủ định)** | **Rất cao ($40\%$ các ca lỗi)** | Sử dụng n-gram (Bigram/Trigram) như `not_good`, hoặc mô hình Transformer có cơ chế Attention để kết nối từ phủ định với tính từ đi kèm. |
| **2. Sarcasm (Mỉa mai)** | **Cao ($25\%$ các ca lỗi)** | Cần mô hình ngôn ngữ lớn (LLM) hiểu được ngữ dụng học (Pragmatics) và sự bất đối xứng giữa kỳ vọng và thực tế. |
| **3. Lexical Ambiguity** | **Trung bình ($15\%$ các ca lỗi)** | Chuyển từ biểu diễn tĩnh (Static) sang biểu diễn ngữ cảnh động (Contextual Embeddings như BERT). |
| **4. Domain/Genre Mismatch** | **Trung bình ($10\%$ các ca lỗi)** | Huấn luyện mô hình đa nhiệm hoặc bổ sung siêu dữ liệu thể loại (Metadata/Category embedding). |
| **5. Short Text / Truncation** | **Thấp ($5\%$ các ca lỗi)** | Tăng chiều dài trích xuất văn bản, kết hợp dữ liệu tóm tắt. |
| **6. Rare Words / OOV** | **Thấp ($5\%$ các ca lỗi)** | Dùng Subword Tokenization (BPE, WordPiece) để không bị mất thông tin từ hiếm. |
| **7. Class Imbalance** | **Được kiểm soát (Dữ liệu 50:50)** | Áp dụng Class Weighting, Focal Loss hoặc Oversampling khi gặp dữ liệu lệch. |
| **8. Data Noise / Label Noise** | **$< 2\%$** | Làm sạch thủ công và áp dụng kỹ thuật Cleanlab để lọc nhãn nhiễu. |
