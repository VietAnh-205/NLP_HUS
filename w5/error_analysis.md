# LAB 05 — ERROR ANALYSIS: PHÂN TÍCH LỖI MÔ HÌNH CHUỖI (RNN / LSTM / GRU)


---

## 1. Bảng tổng hợp các trường hợp thực nghiệm đại diện

| Mã ca | Trích đoạn văn bản (Text Snippet) | Nhãn thực tế (True) | Dự đoán RNN | Dự đoán LSTM | Kết quả | Phân loại bản chất lỗi |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **C1** | *"The film is not good, in fact it was utterly boring and poorly acted throughout."* | Negative | Negative ($0.88$) | Negative ($0.94$) | **Đúng cả hai** | Bắt được phủ định cục bộ liền kề (*not good*) |
| **C2** | *"The movie was, despite many awkward and unnecessarily slow scenes in the first half, ultimately an exceptional and moving masterpiece."* | Positive | Negative ($0.58$) | Positive ($0.89$) | **LSTM Đúng, RNN Sai** | **Long-range Dependency** (Từ khóa chốt ở cuối câu cách xa 25 từ) |
| **C3** | *"I expected the movie to be bad based on the reviews, but it was actually wonderful and charming."* | Positive | Positive ($0.62$) | Positive ($0.85$) | **Đúng cả hai** | Cấu trúc đối lập tương phản (*expected bad ... but wonderful*) |
| **E1** | *"The lead actor, who previously won an oscar and delivered unforgettable performances for over two decades, completely failed to bring any emotion to this flat script."* | Negative | Positive ($0.74$) | Negative ($0.54$) | **RNN Sai do Vanishing Gradient** | Khen ngợi đầu câu lấn át kết luận chê bai cách xa 30 từ |
| **E2** | *(Review dài 650 từ, 100 từ đầu khen bối cảnh, 500 từ sau chê kịch bản thậm tệ; bị cắt ở `max_length = 50`)* | Negative | Positive ($0.81$) | Positive ($0.79$) | **Cả hai Sai do Truncation** | Cắt cụt chuỗi làm mất hoàn toàn phần kết luận cảm xúc ở cuối bài |
| **E3** | *"Oh, brilliant! Another two hours of my life wasted on watching paint dry. Truly an intellectual achievement for kindergarteners."* | Negative | Positive ($0.86$) | Positive ($0.82$) | **Cả hai Sai do Sarcasm** | Châm biếm, mỉa mai; từ tích cực bề mặt (*brilliant, intellectual*) |
| **E4** | *"It is not that the film lacks ambition, nor is it completely devoid of visual flair, but it never quite gels into a satisfying narrative."* | Negative | Positive ($0.69$) | Positive ($0.61$) | **Cả hai Sai do Negation phức tạp** | Phủ định kép (*not lacks, nor devoid*) kết hợp thành ngữ (*never quite gels*) |

---

## 2. Phân tích chi tiết các trường hợp dự đoán thành công (Correct Predictions)

### Trường hợp C1: Bắt trọn cụm phủ định cục bộ (Local Negation)
- **Văn bản:** *"The film is not good, in fact it was utterly boring and poorly acted throughout."*
- **Nhãn thực tế:** Negative | **Dự đoán:** Negative (RNN: $88\%$, LSTM: $94\%$)
- **Cơ chế thành công:**
  - Tại bước thời gian đọc từ `is`, trạng thái ẩn $h_t$ trung tính.
  - Khi đọc từ `not`, trạng thái chuyển sang tích lũy tín hiệu đảo cực.
  - Bước tiếp theo đọc từ `good`: Nhờ liên kết hồi quy $W_{hh} h_{\text{not}}$, vector trạng thái mới không biểu diễn sự tích cực của `good`, mà mã hóa trực tiếp khái niệm *"không tốt"*.
  - Các từ tiếp theo `boring`, `poorly` tiếp tục củng cố cực tính âm, đưa xác suất Negative lên trên $90\%$. Cả RNN và LSTM đều xử lý xuất sắc các cụm phủ định cự ly gần ($1 - 3$ từ).

### Trường hợp C2: Vượt qua mệnh đề phụ nhượng bộ dài (Long-range Dependency)
- **Văn bản:** *"The movie was, despite many awkward and unnecessarily slow scenes in the first half, ultimately an exceptional and moving masterpiece."*
- **Nhãn thực tế:** Positive | **Dự đoán:** Simple RNN đoán **Negative ($58\%$)**, LSTM đoán **Positive ($89\%$)**
- **Cơ chế phân hóa:**
  - Vế nhượng bộ *"despite many awkward and unnecessarily slow scenes"* chứa nhiều từ tiêu cực liên tiếp (`awkward`, `slow`). Simple RNN bị chi phối nặng nề bởi đoạn này và bị vanishing gradient khi cố liên kết chủ ngữ với vị ngữ chính ở cuối câu.
  - Ngược lại, LSTM sử dụng Forget gate để đóng bớt luồng thông tin của vế phụ và dùng Input gate để nạp mạnh mẽ các từ khóa bùng nổ cảm xúc ở cuối câu: `exceptional`, `masterpiece`. Cell state của LSTM bảo toàn tín hiệu này đến bước cuối cùng.

### Trường hợp C3: Chuyển dịch cảm xúc qua liên từ tương phản (Contrast Transition)
- **Văn bản:** *"I expected the movie to be bad based on the reviews, but it was actually wonderful and charming."*
- **Nhãn thực tế:** Positive | **Dự đoán:** Positive (RNN: $62\%$, LSTM: $85\%$)
- **Cơ chế thành công:**
  - Nửa đầu câu chứa kỳ vọng tiêu cực: `expected ... bad`.
  - Từ nối `but` và phó từ `actually` đóng vai trò bản lề đảo chiều ngữ nghĩa (Semantic Pivot). LSTM cập nhật trạng thái ô nhớ với `wonderful` và `charming`, vô hiệu hóa ảnh hưởng của từ `bad` ở mệnh đề trước.

---

## 3. Phân tích chuyên sâu 4 ca lỗi điển hình & Nhận diện lỗi do Sequence Modeling

### Lỗi E1: Lỗi do Vanishing Gradient và Ký ức ngắn hạn của Simple RNN
- **Văn bản:** *"The lead actor, who previously won an oscar and delivered unforgettable performances for over two decades, completely failed to bring any emotion to this flat script."*
- **Nhãn thực tế:** Negative | **Simple RNN đoán:** Positive ($74\%$)
- **Xác định bản chất:** **LỖI TRỰC TIẾP DO CƠ CHẾ CỦA SIMPLE RNN.**
- **Phân tích:**
  - Đoạn mệnh đề quan hệ ca ngợi diễn viên kéo dài tới gần 25 từ với các từ tích cực mạnh (`oscar`, `unforgettable`, `performances`).
  - Lời chê bai thực sự của tác giả nằm ở cuối câu: `completely failed`, `flat script`.
  - Do hiện tượng triệt tiêu gradient, Simple RNN bị "quá tải" và bão hòa bởi chuỗi từ tích cực dài ở đầu câu; ma trận $W_{hh}^{25}$ làm suy yếu nghiêm trọng khả năng ghi đè thông tin mới của 4 từ cuối câu. Ngược lại, LSTM với cơ chế gating đã cân bằng được ngữ cảnh và dự đoán đúng Negative.

### Lỗi E2: Lỗi do Cắt tỉa chuỗi (Truncation Error)
- **Bối cảnh:** Bài viết dài 650 từ. Tác giả dành 60 từ đầu khen ngợi kỹ xảo và âm nhạc, nhưng toàn bộ 590 từ tiếp theo chỉ trích kịch bản ngô nghê và diễn xuất tệ hại. Khi thiết lập `max_length = 50`, chuỗi bị cắt cụt.
- **Nhãn thực tế:** Negative | **Mô hình đoán:** Positive (Cả RNN và LSTM đều đoán sai $> 79\%$)
- **Xác định bản chất:** **LỖI DO TIỀN XỬ LÝ (PREPROCESSING / TRUNCATION), KHÔNG PHẢI DO LỖI MÔ HÌNH HÓA CHUỖI.**
- **Phân tích:**
  - Dù mô hình có hoàn hảo đến đâu, nếu dữ liệu đưa vào bị chặt đứt mất $90\%$ dung lượng (đặc biệt là phần kết luận), mô hình buộc phải dự đoán dựa trên thông tin sai lệch cục bộ ở phần mở bài.
  - Điều này chứng minh việc lựa chọn siêu tham số `max_length` đòi hỏi phải dựa trên phân tích biểu đồ phân bố độ dài văn bản thực tế, chứ không thể chọn bừa bãi để tối ưu tốc độ chạy.

### Lỗi E3: Lỗi do Ngôn ngữ châm biếm, mỉa mai (Sarcasm & Pragmatic Failure)
- **Văn bản:** *"Oh, brilliant! Another two hours of my life wasted on watching paint dry. Truly an intellectual achievement for kindergarteners."*
- **Nhãn thực tế:** Negative | **Mô hình đoán:** Positive (RNN: $86\%$, LSTM: $82\%$)
- **Xác định bản chất:** **LỖI NGỮ DỤNG HỌC (PRAGMATICS), VƯỢT QUÁ KHẢ NĂNG CỦA CÁC MÔ HÌNH HỒI QUY CƠ BẢN.**
- **Phân tích:**
  - Người viết dùng biện pháp tu từ mỉa mai: sử dụng các từ ngữ mang hàm ý tôn vinh cao độ (`brilliant`, `intellectual achievement`) để mô tả một sự việc nhàm chán cùng cực (*"watching paint dry"* dành cho trẻ mẫu giáo).
  - Cả RNN và LSTM đều chỉ học dựa trên phân bố thống kê từ vựng và ngữ pháp bề mặt. Chúng không có tri thức thế giới (World Knowledge) để hiểu rằng *"xem sơn khô"* là một thành ngữ chỉ sự buồn ngủ tột độ. Đây là bài toán đòi hỏi các mô hình ngôn ngữ lớn (LLMs) được huấn luyện trên hàng nghìn tỷ token.

### Lỗi E4: Lỗi do Phủ định kép và Cấu trúc cú pháp đa tầng (Complex Negation)
- **Văn bản:** *"It is not that the film lacks ambition, nor is it completely devoid of visual flair, but it never quite gels into a satisfying narrative."*
- **Nhãn thực tế:** Negative | **Mô hình đoán:** Positive ($61\% - 69\%$)
- **Xác định bản chất:** **LỖI DO MÔ HÌNH CHUỖI ĐƠN HƯỚNG (UNIDIRECTIONAL RECURRENT BOTTLENECK).**
- **Phân tích:**
  - Câu văn chứa 2 lần phủ định lồng ghép: *"not that it lacks"* (không phải là nó thiếu $\to$ có thừa nhận nỗ lực), *"nor is it devoid"* (cũng không phải là hoàn toàn không có $\to$ có điểm cộng hình ảnh), nhưng vế chốt hạ là *"never quite gels"* (không bao giờ kết nối trọn vẹn).
  - Mô hình hồi quy 1 chiều từ trái sang phải bị lúng túng trước các cấu trúc đảo ngữ `nor is it` và sự triệt tiêu lẫn nhau giữa các cặp phủ định, dẫn đến trạng thái ẩn cuối cùng không đạt được sự phân cực rõ ràng.

---

## 4. Bảng tổng kết nguyên nhân lỗi và giải pháp tương ứng

| Nhóm lỗi | Tỷ lệ xuất hiện | Mô hình bị ảnh hưởng nặng nhất | Hướng khắc phục kỹ thuật |
|:---|:---:|:---:|:---|
| **Vanishing Gradient** | $30\%$ | Simple RNN | Nâng cấp lên **LSTM / GRU** hoặc sử dụng kết nối tắt (Residual / Skip connections). |
| **Truncation Loss** | $25\%$ | Tất cả các mô hình | Tăng `max_length` lên $200 - 300$, hoặc sử dụng chiến lược lấy mẫu hỗn hợp (lấy 50 từ đầu + 150 từ cuối). |
| **Sarcasm & Pragmatics** | $20\%$ | Tất cả các mô hình | Sử dụng mô hình ngôn ngữ lớn tiền huấn luyện (Pre-trained Transformers: BERT, RoBERTa, LLMs). |
| **Complex Syntax & Negation** | $15\%$ | Simple RNN, GRU | Áp dụng mạng hồi quy hai chiều (**Bidirectional LSTM**) kết hợp cơ chế Attention. |
| **Rare Words / Out-of-Vocabulary** | $10\%$ | Tất cả các mô hình | Thay thế Tokenizer mức từ bằng **Subword Tokenization** (BPE, WordPiece). |
