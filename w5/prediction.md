# LAB 05 — PREDICTION TRƯỚC EXPERIMENT


---

## Prediction 1 — Xử lý trật tự từ: TF-IDF + Logistic Regression vs RNN + Embedding

* **Câu hỏi:** Giữa mô hình **TF-IDF + Logistic Regression** và **RNN + Embedding**, mô hình nào có khả năng xử lý trật tự từ vựng (Word Order) trực tiếp hơn?
* **Prediction:** **RNN + Embedding** có khả năng xử lý trật tự từ trực tiếp và vượt trội hoàn toàn.
* **Reason:**
  1. **Bản chất của TF-IDF:** Hoàn toàn dựa trên giả thiết túi từ (Bag-of-Words). TF-IDF chỉ đếm tần suất xuất hiện tổng thể của từ trong tài liệu và triệt tiêu từ phổ thông bằng IDF. Nó hoàn toàn bất lực trước trật tự từ: hai câu đảo ngược ý nghĩa như *"dog bites man"* và *"man bites dog"* có vector TF-IDF y hệt nhau.
  2. **Bản chất của RNN:** Xử lý văn bản như một chuỗi thời gian tuần tự $x_1, x_2, \dots, x_T$. Trạng thái ẩn tại mỗi bước $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$ phụ thuộc trực tiếp vào từ hiện tại $x_t$ và toàn bộ lịch sử ngữ cảnh $h_{t-1}$ đi trước nó.
  3. Do phép nhân ma trận $W_{hh} h_{t-1}$ không có tính giao hoán, thứ tự xuất hiện của từ ngữ quyết định trực tiếp đường đi của trạng thái ẩn trong không gian biểu diễn, cho phép RNN phân biệt cấu trúc cú pháp và vị trí của các từ đảo cực như `not`, `never` đi liền trước tính từ.
* **Confidence:** **99%**

---

## Prediction 2 — Khả năng xử lý phụ thuộc xa: Simple RNN vs LSTM

* **Câu hỏi:** Giữa **Simple RNN** và **Long Short-Term Memory (LSTM)**, mô hình nào có khả năng xử lý phụ thuộc tầm xa (Long-range Dependencies) tốt hơn? Giải thích bằng cơ chế cấu trúc, không chỉ nói *"LSTM hiện đại hơn"*.
* **Prediction:** **LSTM** xử lý phụ thuộc tầm xa tốt hơn **Simple RNN** một cách áp đảo.
* **Reason:**
  1. **Hạn chế cố hữu của Simple RNN:** 
     - Quá trình lan truyền ngược qua thời gian (BPTT) đòi hỏi nhân liên tiếp các ma trận Jacobian: $\prod_{k=t+1}^T \text{diag}(1 - h_k^2) W_{hh}^T$.
     - Vì đạo hàm của $\tanh$ luôn $\le 1$, khi khoảng cách $T - t > 10 - 20$ bước, gradient bị suy giảm theo hàm mũ về $0$ (**Vanishing Gradient**). Trọng số mô hình ở các bước đầu câu không nhận được tín hiệu lỗi để cập nhật, khiến Simple RNN bị mất trí nhớ ngữ cảnh xa.
  2. **Giải pháp kiến trúc của LSTM:**
     - LSTM tách biệt biểu diễn thành hai luồng: trạng thái ẩn $h_t$ và trạng thái ô nhớ $c_t$ (Cell State).
     - Phương trình cập nhật ô nhớ mang tính **cộng tính (additive update)**: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$.
     - Đạo hàm $\frac{\partial c_t}{\partial c_{t-1}} = f_t$. Khi cổng quên $f_t \approx 1$, gradient có thể chảy ngược xuyên suốt hàng trăm bước thời gian dọc theo cell state mà không bị nhân với ma trận trọng số hay đạo hàm bão hòa. Đây là "đường cao tốc gradient" (Gradient Highway) bảo toàn thông tin từ xa.
* **Confidence:** **95%**

---

## Prediction 3 — Tác động khi tăng Sequence Length từ 50 lên 200 tokens

* **Câu hỏi:** Nếu độ dài chuỗi cắt tỉa (`max_length`) tăng từ **50 lên 200 tokens**, quá trình huấn luyện và hiệu năng của Simple RNN sẽ thay đổi như thế nào?
* **Prediction:**
  - **Thời gian huấn luyện:** Tăng tuyến tính xấp xỉ gấp $3.5 - 4$ lần.
  - **Bộ nhớ tiêu thụ:** Tăng đáng kể do phải lưu trữ đồ thị tính toán (computation graph) của $200$ bước thời gian để phục vụ BPTT.
  - **Khả năng học của Simple RNN:** Gặp hiện tượng **nghẽn thông tin và suy giảm gradient nghiêm trọng**. Mặc dù $200$ tokens giữ lại được nhiều nội dung hơn, Simple RNN rất khó tận dụng được thông tin ở các vị trí đầu câu do vanishing gradient, dẫn đến việc cải thiện $F_1$-score rất hạn chế so với LSTM.
* **Reason:**
  - Độ phức tạp tính toán của RNN trên một mẫu là $\mathcal{O}(T \cdot d^2)$, phụ thuộc tuyến tính vào chiều dài chuỗi $T$. Tăng $T$ từ $50 \to 200$ làm số lượng phép nhân ma trận tăng gấp 4 lần.
  - Độ dài chuỗi $200$ vượt xa ngưỡng "trí nhớ hiệu dụng" của Simple RNN ($T_{\text{effective}} \approx 10 - 20$), trong khi lại gia tăng số lượng tham số cần lan truyền ngược, khiến quá trình tối ưu hóa bằng Gradient Descent trở nên kém ổn định hơn.
* **Confidence:** **90%**

---

## Prediction 4 — Tác động của việc cắt ngắn văn bản (Truncation) về 30 tokens

* **Câu hỏi:** Nếu cắt toàn bộ bài đánh giá (Review) về chỉ **30 tokens đầu tiên**, thông tin phân loại cảm xúc có bị ảnh hưởng nghiêm trọng không?
* **Prediction:** **CÓ, ảnh hưởng rất tiêu cực**, hiệu năng phân loại ($F_1$ và Accuracy) sẽ sụt giảm mạnh.
* **Reason:**
  1. Trong ngữ liệu phim ảnh (như IMDb / Movie Reviews), độ dài trung bình của một bài viết là khoảng **$700$ từ**.
  2. **Thói quen hành văn phê bình:** Người viết thường mở đầu bài viết bằng việc tóm tắt nội dung kịch bản, giới thiệu dàn diễn viên hoặc dẫn nhập bối cảnh trung tính trong $50 - 100$ từ đầu tiên.
  3. **Kết luận cảm xúc nằm ở phần thân và cuối bài:** Những câu chốt hạ mang tải trọng đánh giá then chốt (*"Overall, this was a complete waste of time"* hoặc *"In conclusion, a brilliant masterpiece"*) thường nằm ở cuối đoạn văn.
  4. Nếu cắt tỉa cụt ngủn ở $30$ tokens, mô hình chỉ đọc được phần giới thiệu mở màn, mất hoàn toàn các từ khóa cảm xúc thực sự, biến bài toán thành dự đoán mù quáng dựa trên ngữ cảnh chưa hoàn chỉnh.
* **Confidence:** **95%**

---

## Prediction 5 — Tính ưu việt của LSTM so với Simple RNN trên mọi tập dữ liệu

* **Câu hỏi:** Mô hình LSTM có **chắc chắn** cho Accuracy cao hơn Simple RNN trên **mọi** tập dữ liệu phân loại văn bản không?
* **Prediction:** **KHÔNG CHẮC CHẮN**, trên một số tập dữ liệu đặc thù, Simple RNN có thể đạt kết quả ngang ngửa hoặc thậm chí cao hơn LSTM.
* **Reason:**
  1. **Số lượng tham số (Parameters Count):**
     - Simple RNN chỉ có $1$ tập trọng số: $W_{xh}, W_{hh}, b_h \implies \mathcal{O}(d \cdot h + h^2)$.
     - LSTM có tới $4$ cổng ($f, i, c, o$), số lượng tham số **gấp 4 lần** Simple RNN: $\mathcal{O}(4(d \cdot h + h^2))$.
  2. **Nguy cơ Overfitting trên tập dữ liệu nhỏ:** Khi tập dữ liệu huấn luyện có kích thước hạn chế (vài trăm mẫu) hoặc văn bản rất ngắn (như tin nhắn SMS, tweet $10 - 20$ từ):
     - Không có sự phụ thuộc xa để LSTM phát huy ưu thế.
     - Dung lượng tham số quá lớn của LSTM dễ dẫn đến hiện tượng học vẹt (Overfitting), trong khi Simple RNN với số tham số gọn nhẹ hơn lại có tính khái quát hóa (Generalization) tốt hơn và hội tụ nhanh hơn.
* **Confidence:** **90%**
