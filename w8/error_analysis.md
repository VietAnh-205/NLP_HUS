
---

## 1. Bảng tổng hợp các trường hợp phân tích lỗi

| Mã ca | Loại lỗi | Câu văn bản đầu vào (Review Text) | True Label | Predicted | Confidence | Nhóm phân loại lỗi | Giả thuyết nguyên nhân (Hypothesis) |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| **FP-1** | False Positive | *"The movie looked visually immaculate with gorgeous lighting, but the story was utterly hollow."* | Negative | **Positive** | $0.684$ | Ambiguous sentiment / Aspect split | Trọng số từ khen ngợi phần nhìn (*"immaculate", "gorgeous"*) kéo lệch vector `[CLS]`, lấn át lời chê cốt truyện. |
| **FP-2** | False Positive | *"I expected a brilliant cinematic triumph, but it turned into an agonizingly tedious slog."* | Negative | **Positive** | $0.725$ | Expectation vs. Reality | Cặp từ tích cực mạnh (*"brilliant triumph"*) ở mệnh đề kỳ vọng quá khứ chi phối không gian ẩn hơn từ chê bai ở vế sau. |
| **FP-3** | False Positive | *"The director made an ambitious attempt to tackle existentialism, yet ultimately delivered a confusing bore."* | Negative | **Positive** | $0.612$ | Context / Ambiguity | Tính từ mang tính khích lệ (*"ambitious attempt"*) được tokenizer và encoder gán mức biểu cảm tích cực, làm mờ sắc thái chê bai. |
| **FN-1** | False Negative | *"Not once in the entire three hours did the director fail to keep my eyes glued to the screen."* | Positive | **Negative** | $0.768$ | Complex negation / Inversion | Cấu trúc đảo ngữ phủ định kép (*"Not once ... fail to ..."*) mang nghĩa khẳng định mạnh mẽ bị nhầm thành tiêu cực do có *"not"* và *"fail"*. |
| **FN-2** | False Negative | *"The plot is admittedly cheesy and cheap, but that is precisely what makes it an absolute cult classic."* | Positive | **Negative** | $0.655$ | Domain-specific expression | Cụm từ *"cult classic"* biểu thị sự yêu thích đặc thù của người mê phim B-movie bị lấn át bởi các từ chê thông thường (*"cheesy", "cheap"*). |
| **FN-3** | False Negative | *"It is dark, depressing, and leaves you emotionally devastated, which is exactly why it is a masterpiece."* | Positive | **Negative** | $0.740$ | Label ambiguity / Sentiment conflict | Trong thể loại phim bi kịch tâm lý, các tính từ đau đớn (*"depressing", "devastated"*) là lời khen nghệ thuật nhưng mô hình diễn giải thành tiêu cực. |
| **HC-1** | **High-Conf Error** | *"Oh, what a masterclass in filmmaking if you enjoy watching paint peel off a wooden fence in total silence."* | Negative | **Positive** | **$0.962$** | Sarcasm / Irony | Lớp vỏ từ ngữ khen ngợi tuyệt đối (*"masterclass in filmmaking"*) kích hoạt xác suất Positive cực cao; mô hình thiếu tri thức thế giới để hiểu ẩn dụ mỉa mai. |
| **HC-2** | **High-Conf Error** | *"The screenwriters certainly did not omit any clichés known to human history."* | Negative | **Positive** | **$0.915$** | Complex negation | Cụm phủ định *"did not omit clichés"* (nghĩa là chứa đầy những hạt sạn rập khuôn) bị vector `[CLS]` nhận diện sai cực tính do từ phủ định *"not"* đi với động từ phủ định *"omit"*. |

---

## 2. Phân tích chuyên sâu các nhóm nguyên nhân chính

### 2.1. Nhóm Lỗi Châm biếm / Mỉa mai (Sarcasm & Irony — Ca HC-1)
- **Câu văn:** *"Oh, what a masterclass in filmmaking if you enjoy watching paint peel off a wooden fence in total silence."*
- **Đặc trưng hành vi:** Mô hình dự đoán nhãn Positive với độ tự tin áp đảo **$96.2\%$** (Confidence cực cao).
- **Phân tích cơ chế:**
  - Token `[CLS]` tiếp nhận sự đóng góp cực mạnh từ cụm danh từ danh giá *"masterclass in filmmaking"*.
  - Mệnh đề điều kiện mang tính ẩn dụ mỉa mai *"watching paint peel off a wooden fence"* sử dụng các từ vựng trung tính (*"paint", "peel", "fence", "silence"*), hoàn toàn không chứa bất kỳ từ khóa tiêu cực rõ rệt nào trong từ điển.
  - **Giới hạn học máy:** BERT chỉ dựa vào tương quan xác suất thống kê đồng xuất hiện bề mặt, hoàn toàn thiếu tri thức thế giới (World Knowledge) và năng lực ngữ dụng học (Pragmatics) để nhận ra hành vi "nhìn sơn bong" là ẩn dụ của sự nhàm chán cùng cực.

### 2.2. Nhóm Lỗi Phủ định kép & Đảo ngữ phức tạp (Complex Negation — Ca FN-1 & HC-2)
- **Câu văn:** *"Not once in the entire three hours did the director fail to keep my eyes glued to the screen."*
- **Đặc trưng hành vi:** Mô hình dự đoán nhãn Negative ($76.8\%$).
- **Phân tích cơ chế:**
  - Trong câu xuất hiện hai token mang sắc thái tiêu cực truyền thống: *"not"* và *"fail"*.
  - Dù BERT có cơ chế Self-Attention hai chiều, việc tổng hợp biểu diễn tại vector `[CLS]` qua các tầng vẫn bị chi phối bởi mật độ từ mang cực tính tiêu cực đứng ở vị trí mở đầu câu.
  - Động từ khẳng định cốt lõi nằm ở cuối chuỗi (*"keep eyes glued"*), khiến cấu trúc đảo ngữ khẳng định (*"luôn luôn cuốn hút"*) bị phân tích thành cấu trúc phủ định thất bại.

### 2.3. Nhóm Lỗi Đa chiều giữa các Khía cạnh (Aspect-Level & Sentiment Ambiguity — Ca FP-1 & FN-3)
- **Câu văn:** *"The movie looked visually immaculate with gorgeous lighting, but the story was utterly hollow."*
- **Đặc trưng hành vi:** Dự đoán Positive ($68.4\%$).
- **Phân tích cơ chế:**
  - Người đánh giá khen khía cạnh thị giác (*"visually immaculate", "gorgeous lighting"*) nhưng chê khía cạnh kịch bản (*"story utterly hollow"*).
  - Vì phân loại cấp văn bản (Document-level Classification) chỉ gán một nhãn duy nhất cho toàn bộ bài review, vector `[CLS]` phải nén đồng thời cả hai luồng cảm xúc đối nghịch.
  - Các tính từ khen ngợi thị giác có trọng số chú ý cao hơn, làm lệch vector đại diện sang nhãn tích cực, dù kết luận chung của người xem là thất vọng.

---

## 3. Tổng kết bài học từ Error Analysis

1. **Confidence $\neq$ Correctness:** Hai ca lỗi **HC-1** ($96.2\%$) và **HC-2** ($91.5\%$) chứng minh rằng mô hình hoàn toàn có thể cực kỳ tự tin (Overconfident) vào các quyết định sai lầm khi câu văn bị ngụy trang bằng từ ngữ tích cực bề mặt hoặc cấu trúc đảo ngữ phức tạp.
2. **Nhu cầu về Aspect-based Sentiment Analysis:** Phân loại cảm xúc nhị phân toàn văn bản có giới hạn cố hữu khi đối mặt với các bài đánh giá đa chiều (vừa khen vừa chê).
3. **Giới hạn của Pretrained Representation:** Tiền huấn luyện trên Wikipedia và Books giúp BERT hiểu cú pháp chuẩn, nhưng không tự động giúp mô hình nắm bắt được sự mỉa mai hay tiếng lóng của văn hóa đại chúng nếu không có dữ liệu tinh chỉnh chuyên biệt.

