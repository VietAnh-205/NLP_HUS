# LAB 04 — PREDICTION TRƯỚC EXPERIMENT


## Prediction 1 — Hiệu năng của TF-IDF kết hợp Logistic Regression

* **Câu hỏi:** Mô hình baseline gồm biểu diễn **TF-IDF kết hợp Logistic Regression** có khả năng hoạt động tốt trên bài toán phân loại văn bản không?
* **Prediction:** **CÓ, hoạt động rất tốt** và là một baseline cực kỳ mạnh mẽ (strong baseline), rất khó bị đánh bại bởi các mô hình đơn giản khác.
* **Reason:**
  1. Trong các bài toán phân loại văn bản theo chủ đề hoặc cảm xúc, nhãn thường được quyết định mạnh mẽ bởi sự hiện diện của một tập hợp các từ khóa phân biệt (ví dụ: các từ cảm xúc `excellent`, `masterpiece` vs `terrible`, `worst` trong sentiment; hoặc `cpu`, `software` vs `orbit`, `space` trong topic).
  2. TF-IDF có cơ chế kép tối ưu: Term Frequency (TF) đo lường mức độ xuất hiện cục bộ, còn Inverse Document Frequency (IDF) triệt tiêu trọng số của các từ phổ thông, làm nổi bật chính xác các từ khóa đặc trưng.
  3. Logistic Regression hoạt động cực kỳ hiệu quả trên không gian đặc trưng thưa nhiều chiều (High-dimensional Sparse Space): nó học trực tiếp các trọng số $w_i > 0$ cho từ khóa tích cực và $w_i < 0$ cho từ khóa tiêu cực, tạo ra một siêu phẳng phân cách tuyến tính chuẩn xác với tốc độ huấn luyện nhanh và ít bị overfit khi có điều chuẩn L2.
* **Confidence:** **95%**

---

## Prediction 2 — So sánh Average Word2Vec và TF-IDF

* **Câu hỏi:** Biểu diễn văn bản bằng véc-tơ trung bình Word2Vec (Average Word2Vec) có chắc chắn tốt hơn biểu diễn TF-IDF không?
* **Prediction:** **KHÔNG CHẮC CHẮN**, thậm chí Average Word2Vec có khả năng cho kết quả **kém hơn TF-IDF** trên bài toán phân loại văn bản (đặc biệt là phân loại cảm xúc).
* **Reason:**
  1. **Hiệu ứng pha loãng thông tin (Information Dilution):** Khi lấy trung bình cộng tất cả các word vectors trong văn bản $\vec{x}_{\text{doc}} = \frac{1}{n} \sum_{i=1}^n \vec{w}_i$, các từ khóa quyết định cảm xúc (chỉ chiếm vài từ) sẽ bị "hòa tan" và lấn át bởi hàng trăm từ nền trung tính (danh từ, động từ mô tả cảnh vật, nhân vật).
  2. **Mất hoàn toàn thứ tự từ (Loss of Word Order):** Cả hai mô hình đều là Bag-of-Words, nhưng TF-IDF giữ nguyên tính độc lập của từng chiều từ vựng (orthogonal features), trong khi Average Word2Vec gom tất cả vào một véc-tơ đặc số chiều thấp ($d \in [50, 300]$).
  3. **Vấn đề từ trái nghĩa trong không gian nhúng:** Trong Word2Vec, các từ trái nghĩa như `good` và `bad` thường xuất hiện trong cùng ngữ cảnh (ví dụ: *"the movie is very..."*), do đó khoảng cách Cosine giữa chúng rất gần nhau. Khi lấy trung bình vector, mô hình phân loại tuyến tính rất khó tìm được ranh giới phân tách sắc bén như khi dùng các chiều từ vựng độc lập của TF-IDF.
* **Confidence:** **85%**

---

## Prediction 3 — Tác động của Kích thước Từ vựng (Vocabulary Size)

* **Câu hỏi:** Nếu tăng kích thước từ vựng (Vocabulary Size) bằng cách giữ lại toàn bộ các từ hiếm thì Accuracy của mô hình có chắc chắn tăng không?
* **Prediction:** **KHÔNG**, Accuracy sẽ **không chắc chắn tăng**, thậm chí sẽ giảm hoặc nhanh chóng đạt ngưỡng bão hòa.
* **Reason:**
  1. Khi tăng vocabulary không kiểm soát (giữ lại cả các từ xuất hiện 1 lần — hapax legomena, từ gõ sai chính tả, số ngẫu nhiên, mã HTML), số chiều của không gian đặc trưng bùng nổ, làm tăng độ thưa thớt (sparsity).
  2. **Hiện tượng Overfitting trên từ hiếm:** Các từ hiếm chỉ xuất hiện 1 lần ngẫu nhiên trong một tài liệu của lớp Positive ở tập Train. Mô hình Logistic Regression có thể gán trọng số $w_i$ rất lớn cho từ hiếm này để ép đúng nhãn trên tập Train (học vẹt nhiễu), nhưng từ này sẽ không bao giờ xuất hiện lại hoặc xuất hiện ở ngữ cảnh hoàn toàn khác trong tập Test, dẫn đến giảm khả năng tổng quát hóa (Generalization Gap).
  3. Giới hạn từ vựng thông qua `min_df` hoặc `max_features` đóng vai trò như một bộ lọc nhiễu tự nhiên, giúp mô hình tập trung vào các tín hiệu thống kê đáng tin cậy.
* **Confidence:** **90%**

---

## Prediction 4 — Tác động của việc Loại bỏ Từ dừng (Stopwords Removal)

* **Câu hỏi:** Nếu loại bỏ hoàn toàn các từ dừng (Stopwords) thì hiệu năng phân loại của mô hình có chắc chắn tăng không?
* **Prediction:** **KHÔNG CHẮC CHẮN TĂNG**, phụ thuộc chặt chẽ vào bài toán cụ thể và cách định nghĩa danh sách stopwords. Trong bài toán phân loại cảm xúc (Sentiment Analysis), việc loại bỏ stopwords chuẩn có thể làm **giảm hiệu năng**.
* **Reason:**
  1. Nhiều từ nằm trong danh sách stopwords mặc định (như `not`, `no`, `never`, `nor`, `but`, `very`) lại là các từ mang tải trọng ngữ nghĩa cực kỳ quan trọng:
     - Từ phủ định (`not`, `never`): Có khả năng đảo ngược hoàn toàn cực tính cảm xúc (*"good"* $\to$ *"not good"*).
     - Từ khuếch đại (`very`, `extremely`): Tăng cường mức độ cảm xúc.
     - Liên từ tương phản (`but`, `however`): Báo hiệu mệnh đề chính mang cảm xúc thật sự nằm ở phía sau (*"The actors were fine, but the plot was terrible"*).
  2. Nếu loại bỏ vô tội vạ các từ này, câu *"this movie is not good"* sẽ bị biến thành *"movie good"*, khiến mô hình dự đoán sai hoàn toàn.
  3. Đối với biểu diễn TF-IDF, thuật toán IDF vốn dĩ đã tự động gán trọng số cực thấp cho các từ chức năng phổ biến khắp corpus, nên việc loại bỏ stopwords thủ công thường không mang lại cải thiện đột phá.
* **Confidence:** **90%**

---

## Prediction 5 — Mâu thuẫn giữa Accuracy và F1-score

* **Câu hỏi:** Chỉ số Accuracy và $F_1$-score có thể đưa ra kết luận trái ngược nhau về chất lượng của hai mô hình không?
* **Prediction:** **HOÀN TOÀN CÓ THỂ**, đặc biệt xảy ra khi tập dữ liệu bị mất cân bằng lớp (Class Imbalance).
* **Reason:**
  1. **Bản chất của Accuracy:** Đếm tỷ lệ đoán đúng trên toàn bộ tập dữ liệu, bị chi phối áp đảo bởi lớp chiếm đa số (Majority Class).
  2. **Bản chất của $F_1$-score (đặc biệt là Macro $F_1$):** Là trung bình điều hòa giữa Precision và Recall, đánh giá công bằng chất lượng dự đoán trên từng lớp, đặc biệt nhạy cảm với hiệu năng trên lớp thiểu số (Minority Class).
  3. **Kịch bản thực tế:**
     - Mô hình A thiên vị lớp đa số: Đạt Accuracy $92\%$ nhưng Recall lớp thiểu số chỉ đạt $10\% \implies$ Macro $F_1 \approx 0.52$.
     - Mô hình B cân bằng: Đạt Accuracy $88\%$ (thấp hơn A) nhưng Recall lớp thiểu số đạt $80\%$ và Precision ổn định $\implies$ Macro $F_1 \approx 0.85$.
     - Khi đó: Nếu nhìn theo Accuracy thì kết luận Mô hình A tốt hơn, nhưng nếu nhìn theo $F_1$ thì Mô hình B vượt trội hoàn toàn.
* **Confidence:** **98%**
