# LAB 03 — Word Embeddings & Distributed Representations

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng | Học kỳ I - 2026  
**Đơn vị:** Khoa Công nghệ Thông tin — Trường ĐH Khoa học Tự nhiên, ĐHQGHN  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Sinh viên thực hiện:** Nguyễn Việt Anh  

---

## 1. Tổng quan Lab

Buổi lab này hướng tới việc chuyển hóa các khái niệm lý thuyết về **Biểu diễn từ phân tán (Distributed Word Representations)**, **Ma trận đồng xuất hiện (Co-occurrence Matrix)**, và **Mô hình Word2Vec (CBOW & Skip-gram)** thành kỹ năng thực hành và nghiên cứu chuyên sâu. Bạn sẽ đi qua một chu trình thực nghiệm toàn diện từ Mục 10 đến Mục 30:

> Tính toán tay & Thiết lập giả thuyết → Code thuần túy không phụ thuộc thư viện → Huấn luyện Word2Vec → Khảo sát siêu tham số (Window & Dimension) → Đánh giá Tương đồng từ vựng & Phép loại suy (Analogy) → Ứng dụng Semantic Search → Phân tích lỗi định tính & Khảo sát từ đa nghĩa (Polysemy) → Suy ngẫm tiến hóa biểu diễn ngôn ngữ.

### Mục tiêu cốt lõi:
1. **Hiểu rõ luồng biểu diễn ngôn ngữ:** $\text{Context} \to \text{Co-occurrence} \to \text{Vector} \to \text{Similarity}$.
2. **Khắc phục nút thắt biểu diễn thưa (Sparse bottleneck):** Nhận diện vấn đề bùng nổ chiều không gian khi $|V| = 100,000$ (ma trận $100,000 \times 100,000$ tốn $40 \text{ GB}$ RAM) và lý giải động lực chuyển dịch sang không gian véc-tơ đặc số chiều thấp (Low-dimensional dense embeddings: $d \in [50, 300]$).
3. **Làm chủ hai kiến trúc kinh điển:** Phân biệt cơ chế đầu vào/đầu ra, tốc độ học và khả năng học từ hiếm giữa **Continuous Bag-of-Words (CBOW)** và **Skip-gram**.
4. **Nghiên cứu đánh đổi siêu tham số (Hyperparameter Trade-offs):**
   - Kích thước cửa sổ (Context window $w \in \{2, 5, 10\}$): Đánh đổi giữa quan hệ cú pháp / từ thay thế được (Paradigmatic) và quan hệ liên tưởng chủ đề rộng (Syntagmatic/Topical).
   - Số chiều nhúng ($d \in \{50, 100, 300\}$): Đánh đổi giữa dung lượng bộ nhớ, thời gian huấn luyện và năng lực biểu diễn thông tin.
5. **Đánh giá hình học & Đại số vector:** Đo lường tương đồng từ vựng bằng Cosine similarity và kiểm chứng tính chất tuyến tính của không gian vector thông qua bài toán Analogy ($king - man + woman \approx queen$).
6. **Ứng dụng Semantic Search:** Kết nối với LAB 01 để giải quyết triệt để lỗi không khớp từ vựng (Lexical Mismatch) của TF-IDF bằng cách tìm kiếm ngữ nghĩa theo véc-tơ nhúng trung bình.
7. **Nhận thức giới hạn biểu diễn tĩnh (Static Embeddings):** Phân tích sự sụp đổ không gian (Polysemy collapse) khi từ đa nghĩa như `bank` bị ép vào một vector duy nhất, mở ra tiền đề cho các mô hình ngữ cảnh động như Transformer (BERT/GPT).

---

## 2. Cấu trúc thư mục nộp bài (Deliverables)

Thư mục `w3/` bao gồm đầy đủ 8 tệp tin theo đúng chuẩn quy định tại **Mục 30. Deliverables**:

```
w3/
├── README.md             # Báo cáo tổng quan, kết quả thực nghiệm và hướng dẫn sử dụng
├── calculations.md       # Tính toán chi tiết phép toán Analogy và phân tích dung lượng bộ nhớ
├── prediction.md         # Phân tích mẫu huấn luyện CBOW vs Skip-gram và thiết lập giả thuyết
├── cooccurrence.py       # Module mã nguồn thuần triển khai Co-occurrence Matrix & Similarity
├── word_embedding.ipynb  # Notebook thực nghiệm hoàn chỉnh từ Mục 10 đến Mục 27
├── results.csv           # Dữ liệu định lượng kết quả đo lường tương đồng từ vựng
├── error_analysis.md     # Bảng phân tích chuyên sâu 3 trường hợp đúng và 3 trường hợp sai/bất ngờ
└── reflection.md         # Bảng so sánh 4 thế hệ biểu diễn và trả lời 6 câu hỏi vấn đáp
```

### Mô tả chi tiết từng file:
- **`cooccurrence.py`**: Triển khai thuần túy bằng Python và NumPy (không comment trong code, không dùng thư viện ngoài) các hàm: `build_vocabulary()`, `build_cooccurrence_matrix()`, `cosine_similarity()`, `most_similar()`, `word_similarity()`, `compute_matrix_stats()`.
- **`word_embedding.ipynb`**: Notebook chia cell mạch lạc, chạy được từ đầu đến cuối, gồm toàn bộ các thí nghiệm từ Co-occurrence đến Word2Vec (SGNS tự code thuần NumPy), Word Similarity, Word Analogy, Semantic Search, và Polysemy test.
- **`results.csv`**: File CSV lưu trữ bảng số liệu thực nghiệm định lượng của 6 cặp từ vựng bắt buộc qua 3 cửa sổ ($w=2, 5, 10$) và 3 số chiều ($d=50, 100, 300$).
- **`calculations.md`**: Giải chi tiết bài tập tính vector $king - man + woman$, tính chuẩn L2, Cosine similarity, và chứng minh bài toán 40 GB RAM của ma trận $100,000 \times 100,000$.
- **`prediction.md`**: Liệt kê 4 mẫu huấn luyện CBOW và 6 cặp Skip-gram cho câu *"the cat eats fish"* với $w=1$, kèm phân tích so sánh cơ chế học.
- **`error_analysis.md`**: Phân tích định tính theo 4 mục chuẩn (`Observed`, `Expected`, `Possible explanation`, `Evidence from corpus`) cho 3 cặp đúng (`doctor-nurse`, `king-queen`, `cat-dog`) và 3 trường hợp bất ngờ (`computer-banana`, `doctor-disease`, hiện tượng đa nghĩa của `bank`).
- **`reflection.md`**: Bảng so sánh 4 thế hệ biểu diễn (TF-IDF, Co-occurrence, Word2Vec, Contextual Embeddings), giải thích chuyên sâu bài toán `bank`, và trả lời đầy đủ 6 câu hỏi kiểm tra vấn đáp cá nhân.

---
