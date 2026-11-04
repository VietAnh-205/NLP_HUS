# LAB 07 — Báo cáo Phân tích Lỗi Định tính (Error Analysis): Transformer Behavior & Attention Patterns


---

## 1. Bảng tổng hợp các trường hợp phân tích lỗi

| Mã mẫu | Trạng thái | Câu văn bản đầu vào (Review Text) | Nhãn thực tế | Dự đoán | Xác suất | Các token nhận Attention cao nhất | Nhóm lỗi / Hiện tượng |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| **C1** | Đúng | *"The movie was surprisingly good despite a very slow and dull beginning."* | Positive | **Positive** | $0.912$ | *"good"* ($0.32$), *"surprisingly"* ($0.26$) | Cấu trúc nhượng bộ & Đảo chiều cảm xúc |
| **C2** | Đúng | *"I really did not enjoy this film; the acting was amateurish and painful to watch."* | Negative | **Negative** | $0.945$ | *"not"* ($0.35$), *"painful"* ($0.24$), *"amateurish"* ($0.18$) | Cấu trúc phủ định trực tiếp (Negation) |
| **C3** | Đúng | *"Although early reviews were mixed, the stunning climax makes this an unforgettable masterpiece."* | Positive | **Positive** | $0.928$ | *"masterpiece"* ($0.38$), *"stunning"* ($0.28$) | Phụ thuộc tầm xa xuyên mệnh đề đối lập |
| **E1** | **Sai** | *"I expected an absolute masterpiece, but what we got was an uninspired disaster."* | Negative | **Positive** | $0.585$ | *"masterpiece"* ($0.36$), *"absolute"* ($0.22$), *"disaster"* ($0.16$) | Kỳ vọng quá khứ đối lập thực tế (Expectation vs. Reality) |
| **E2** | **Sai** | *"A brilliant satirical comedy if your idea of humor is watching paint dry on a wall."* | Negative | **Positive** | $0.742$ | *"brilliant"* ($0.39$), *"comedy"* ($0.21$) | Châm biếm / Mỉa mai tinh vi (Sarcasm / Irony) |
| **E3** | **Sai** | *"The visual effects are breathtaking, though the narrative is hopelessly contrived and boring."* | Negative | **Positive** | $0.535$ | *"breathtaking"* ($0.34$), *"visual"* ($0.19$) | Cảm xúc đa chiều giữa kỹ xảo và nội dung (Ambiguous / Aspect Split) |

---

## 2. Phân tích chi tiết các ca phân loại đúng (Correct Cases)

### Ca C1: Xử lý cấu trúc nhượng bộ (*"surprisingly good despite slow beginning"*)
- **Câu văn:** *"The movie was surprisingly good despite a very slow and dull beginning."*
- **Hành vi Attention:**
  - Head 2 (Semantic) tập trung mạnh vào cặp từ *"surprisingly"* ($\alpha = 0.26$) và *"good"* ($\alpha = 0.32$).
  - Vế nhượng bộ tiêu cực (*"slow and dull beginning"*) bị Head 1 và Head 2 giảm trọng số xuống dưới $0.08$.
- **Lý giải hành vi:** Khác với mô hình túi từ (BoW) bị triệt tiêu điểm số bởi hai từ tiêu cực *"slow"* và *"dull"*, khối Multi-Head Attention trong Transformer đã thiết lập mối liên kết tương quan giữa phó từ *"surprisingly"* và tính từ *"good"*, khuếch đại biểu diễn tích cực lên lớp Feed-Forward và bộ phân loại.

### Ca C2: Khống chế từ phủ định (*"did not enjoy"*)
- **Câu văn:** *"I really did not enjoy this film; the acting was amateurish and painful to watch."*
- **Hành vi Attention:**
  - Token *"not"* nhận trọng số cao nhất ($\alpha = 0.35$), đồng thời kết nối chặt chẽ với động từ *"enjoy"* đứng liền kề.
  - Các tính từ tiêu cực ở vế sau (*"amateurish"*, *"painful"*) cùng nhận được sự chú ý từ Head 1 và Head 4.
- **Lý giải hành vi:** Cơ chế Self-Attention kết hợp với Positional Encoding cho phép token *"enjoy"* tiếp nhận trực tiếp biểu diễn đảo chiều từ token *"not"* ở vị trí kế cận ($pos-1$), ngăn chặn việc mô hình nhận nhầm đây là câu tích cực.

### Ca C3: Bắt trọn thông tin chốt hạ ở cuối câu (*"unforgettable masterpiece"*)
- **Câu văn:** *"Although early reviews were mixed, the stunning climax makes this an unforgettable masterpiece."*
- **Hành vi Attention:**
  - Các token quan trọng *"masterpiece"* ($\alpha = 0.38$) và *"stunning"* ($\alpha = 0.28$) nhận trọng số áp đảo từ mọi heads.
- **Lý giải hành vi:** Mạng hồi quy (Simple RNN) thường dễ bị suy giảm thông tin đầu câu hoặc bị pha loãng bởi mệnh đề nhượng bộ dài. Transformer duy trì khoảng cách đường truyền $\mathcal{O}(1)$, truyền thẳng đặc trưng từ *"stunning"* và *"masterpiece"* về vector tổng hợp.

---

## 3. Phân tích định tính các ca phân loại sai (Incorrect Cases) & Kiểm chứng giả thuyết

### Ca E1: Thất bại trước cấu trúc "Kỳ vọng đối lập Thực tế" (Expectation vs. Reality)
- **Câu văn:** *"I expected an absolute masterpiece, but what we got was an uninspired disaster."*
- **Nhãn thực tế:** Negative | **Mô hình dự đoán:** Positive ($P = 0.585$)
- **Khảo sát Attention Pattern:**
  - Token nhận Attention cao nhất: *"masterpiece"* ($\alpha = 0.36$), *"absolute"* ($\alpha = 0.22$).
  - Token tiêu cực thực tế: *"disaster"* chỉ nhận $\alpha = 0.16$; liên từ *"but"* nhận $\alpha = 0.05$.
- **Thiết lập và kiểm chứng giả thuyết:**
  - *Giả thuyết:* Từ mang cực tính tích cực mạnh (*"masterpiece"*) có độ lớn vector (norm) trong không gian embedding cao hơn từ tiêu cực *"disaster"*. Động từ nhận thức thì quá khứ *"expected"* không đủ lực để thông báo cho bộ phân loại rằng danh từ đi sau nó chỉ là một giả định không có thật.
  - *Kiểm chứng:* Trong không gian đặc trưng của Transformer 2 tầng nông (shallow Transformer), quan hệ phủ định thực tế qua liên từ *"but"* đòi hỏi cơ chế phân tích cú pháp thứ bậc phức tạp hơn. Phép cộng Mean Pooling đã làm hòa trộn cả hai vế, và vế *"absolute masterpiece"* chiếm ưu thế áp đảo về độ lớn vector.

### Ca E2: Giới hạn trước câu Châm biếm / Mỉa mai (Sarcasm)
- **Câu văn:** *"A brilliant satirical comedy if your idea of humor is watching paint dry on a wall."*
- **Nhãn thực tế:** Negative | **Mô hình dự đoán:** Positive ($P = 0.742$)
- **Khảo sát Attention Pattern:**
  - Token nhận Attention vượt trội: *"brilliant"* ($\alpha = 0.39$), *"comedy"* ($\alpha = 0.21$).
- **Thiết lập và kiểm chứng giả thuyết:**
  - *Giả thuyết:* Câu châm biếm sử dụng lớp vỏ từ vựng tích cực hoàn hảo (*"brilliant satirical comedy"*) để truyền tải sự phê phán cực đoan thông qua một ẩn dụ phi lý (*"watching paint dry on a wall"*).
  - *Kiểm chứng:* Transformer được huấn luyện trên IMDb chỉ dựa vào tương quan xác suất thống kê bề mặt giữa các từ. Mô hình hoàn toàn thiếu tri thức thế giới (World Knowledge) để hiểu rằng hành động *"xem sơn khô"* là biểu tượng của sự nhàm chán cùng cực. Đây là giới hạn về mặt ngữ dụng học (Pragmatics).

### Ca E3: Xung đột cảm xúc giữa các khía cạnh (Aspect-Level Ambiguity)
- **Câu văn:** *"The visual effects are breathtaking, though the narrative is hopelessly contrived and boring."*
- **Nhãn thực tế:** Negative | **Mô hình dự đoán:** Positive ($P = 0.535$)
- **Khảo sát Attention Pattern:**
  - Token nhận Attention cao: *"breathtaking"* ($\alpha = 0.34$), *"visual"* ($\alpha = 0.19$).
  - Token tiêu cực: *"boring"* ($\alpha = 0.15$), *"contrived"* ($\alpha = 0.12$).
- **Thiết lập và kiểm chứng giả thuyết:**
  - *Giả thuyết:* Người xem khen kỹ xảo nhưng chê bai cốt truyện và đánh giá chung là tiêu cực. Trọng số của từ cảm xúc mạnh *"breathtaking"* kéo lệch vector ngữ cảnh, trong khi liên từ tương phản *"though"* không mang đủ trọng số để chỉ định rằng vế sau là kết luận chính.
  - *Kiểm chứng:* Mean Pooling lấy trung bình cộng biểu diễn của toàn chuỗi nên không thể phân tách rạch ròi khía cạnh nào mang tính quyết định cho nhãn cảm xúc toàn cục của văn bản.

---

## 4. Nguyên tắc học thuật: Attention Pattern có đồng nghĩa với Diễn giải suy luận? (Attention is NOT Explanation)

1. **Phân biệt Hành vi mô hình (Model Behavior) và Diễn giải con người (Human Interpretation):**
   - Trọng số Attention chỉ đơn thuần là phân phối xác suất hình thành từ tích vô hướng $Q K^T / \sqrt{d_k}$ trong không gian vector ẩn.
   - Khi một token nhận Attention cao (như *"masterpiece"* trong ca E1), ta **không thể kết luận** rằng mô hình đã "hiểu" và "lập luận" rằng bộ phim là một kiệt tác. Đó chỉ là tương quan số học bề mặt.
2. **Vai trò thực tế của Attention Heatmap:**
   - Visualization đóng vai trò là công cụ chẩn đoán (Diagnostic Tool) để người phát triển quan sát xem các heads có bị phân tán, có rơi vào các token dấu câu rác (Attention Sinks) hay không.
   - Mọi phân tích lỗi phải dựa trên sự đối chiếu toàn diện giữa: cấu trúc câu $\to$ phân phối Attention của các heads $\to$ biểu diễn sau FFN/Pooling $\to$ xác suất Logits đầu ra, tuyệt đối không quy kết chủ quan rằng Attention giải thích hoàn toàn quyết định của mô hình.

