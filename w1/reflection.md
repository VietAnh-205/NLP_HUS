# REFLECTION

---

**1. Prediction nào của em sai?**
Dự đoán của em về độ thưa (sparsity) của ma trận TF-IDF đã sai. Ban đầu em cho rằng một corpus 30.000 văn bản sẽ có nhiều từ lặp lại, nên ma trận sẽ "đặc" hơn (sparsity khoảng 80-90%). Tuy nhiên, thực tế chứng minh ma trận có độ thưa lên tới hơn 99%. Nguyên nhân là do Vocabulary tổng cực kỳ lớn (hàng chục nghìn từ do chứa cả số, lỗi chính tả, từ hiếm), nhưng mỗi văn bản lại chỉ chứa một lượng rất nhỏ các từ vựng nhất định.

**2. Kết quả nào bất ngờ nhất?**
Điều bất ngờ nhất là sự khác biệt giữa implementation tính toán thủ công và thư viện `scikit-learn`. Em không ngờ rằng `TfidfVectorizer` mặc định áp dụng thêm smoothing (cộng 1 vào tử và mẫu của IDF) và đặc biệt là ngầm thực hiện L2 Normalization (ép độ dài vector về 1). Điều này cho thấy các thư viện thực tế đã thêm vào rất nhiều "trick" tối ưu toán học so với công thức lý thuyết nguyên bản nhằm cải thiện hiệu năng tìm kiếm bằng Cosine Similarity.

**3. Experiment nào cung cấp evidence mạnh nhất?**
Experiment 2 (Preprocessing Ablation) cung cấp evidence mạnh mẽ nhất. Bằng cách so sánh trực tiếp ba pipeline (Minimal, Normalized, Subword) qua các chỉ số định lượng như Vocab Size và OOV Rate, thí nghiệm này chứng minh rõ ràng luận điểm: tiền xử lý không đơn thuần là "dọn rác" dữ liệu, mà là một quyết định mô hình hóa (modeling decision). Việc chọn bỏ dấu câu hay dùng thuật toán subword làm thay đổi hoàn toàn cách văn bản được biểu diễn thành vector.

**4. Failure case quan trọng nhất là gì?**
Đó là sự cố không khớp từ vựng (Lexical Mismatch). Một truy vấn như `"heart attack treatment"` hoàn toàn "mù" trước một văn bản chứa nội dung `"myocardial infarction therapy"`. TF-IDF coi đây là 4 từ vựng trực giao (orthogonal) nằm ở 4 chiều không gian khác nhau, dẫn đến tích vô hướng bằng 0. Hệ thống thất bại vì nó chỉ so sánh chuỗi ký tự (lexical) chứ không hề nắm bắt được sự tương đồng về ngữ nghĩa (semantic).

**5. Nếu được xây lại search engine, em sẽ thay đổi điều gì?**
Nếu vẫn bị giới hạn ở Sparse Representation, em sẽ bổ sung thêm cấu hình N-grams (ví dụ: bi-grams) thay vì chỉ dùng unigram để giữ lại một phần thông tin về trật tự từ (giúp phân biệt `"machine learning"` và `"learning machine"`). Đồng thời, em sẽ thêm bước Lemmatization để đưa các biến thể từ (ví dụ: *treatment, treatments*) về chung một gốc, giúp tăng chỉ số Recall.

**6. AI đã được sử dụng ở những phần nào và đóng góp cụ thể là gì?**
- **Đóng góp của AI:** Giải thích sự khác biệt cơ bản giữa mã nguồn thủ công và cơ chế hoạt động ngầm của `scikit-learn`; phát hiện và sửa lỗi truyền sai tham số giữa `TfidfVectorizer` và `CountVectorizer`; hỗ trợ định dạng kết quả thực nghiệm thành bảng Markdown và DataFrame.
- **Tuân thủ quy tắc:** Em đã tự thực hiện tính toán ở Part B, ghi nhận dự đoán cá nhân ở Part C, và tự tư duy thiết kế các truy vấn học thuật cho tập đánh giá trước khi dùng AI hỗ trợ viết script thống kê.