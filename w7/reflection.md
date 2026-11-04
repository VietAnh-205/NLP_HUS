# LAB 07 — Báo cáo Tổng kết & Suy ngẫm (Reflection): Từ Transformer đến Kỷ nguyên Mô hình Ngôn ngữ Lớn


---

## 1. Bảng so sánh tiến hóa biểu diễn ngôn ngữ qua các buổi Lab (LAB 04 → LAB 07)

| Tiêu chí đặc trưng | TF-IDF (LAB 04) | Avg Word2Vec (LAB 04) | LSTM (LAB 05) | Self-Attention (LAB 06) | Transformer (LAB 07) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dạng biểu diễn vector** | Thưa (Sparse) | Dày đặc (Dense) | Dày đặc (Dense) | Dày đặc (Dense) | Dày đặc (Dense) |
| **Nhạy bén với trật tự từ (Word Order)** | Rất hạn chế (n-gram) | Hoàn toàn không ($\times$) | Tự nhiên theo thời gian ($\checkmark$) | Không ($\times$, mất trật tự) | **Nhạy bén nhờ Positional Encoding ($\checkmark$)** |
| **Biểu diễn phụ thuộc ngữ cảnh (Contextual)** | Không ($\times$) | Không ($\times$) | Động theo bước thời gian ($\checkmark$) | Động theo tích vô hướng ($\checkmark$) | **Động đa chiều qua Multi-Head Attention ($\checkmark$)** |
| **Tương tác trực tiếp giữa các token (Path Length)** | Không có | Không có | Gián tiếp qua $T$ bước ($\mathcal{O}(T)$) | Trực tiếp $\mathcal{O}(1)$ | **Trực tiếp $\mathcal{O}(1)$ qua nhiều không gian con** |
| **Tính toán tuần tự (Sequential Computation)** | Không | Không | Bắt buộc ($\vec{h}_{t-1} \to \vec{h}_t$) | Không bắt buộc | **Hoàn toàn song song hóa trên GPU** |
| **Biến đổi phi tuyến từng vị trí (Position-wise)** | Tuyến tính | Tuyến tính | Phi tuyến qua cổng | Không có (chỉ tổ hợp tuyến tính) | **Có (Feed-Forward Network tại mỗi vị trí)** |
| **Ổn định gradient khi mạng sâu** | Không áp dụng | Không áp dụng | Cải thiện nhờ Cell State $c_t$ | Không áp dụng | **Tối ưu hóa nhờ Residual + LayerNorm** |
| **Chi phí tính toán theo độ dài chuỗi ($T$)** | Tuyến tính ($\mathcal{O}(V)$) | Tuyến tính ($\mathcal{O}(T \cdot d)$) | Tuyến tính ($\mathcal{O}(T \cdot d^2)$) | Bậc hai ($\mathcal{O}(T^2 \cdot d)$) | **Bậc hai ($\mathcal{O}(T^2 \cdot d + T \cdot d_{ff})$)** |

---

## 2. Chuỗi giải quyết giới hạn từ LAB 01 đến LAB 07

Quá trình phát triển biểu diễn ngôn ngữ là một chuỗi liên tục hóa giải các giới hạn cố hữu của thế hệ đi trước:

```
TF-IDF (LAB 01 & 04)
  │
  ▼ Giới hạn: Vector thưa, số chiều khổng lồ bằng kích thước từ vựng |V|, hoàn toàn thiếu ngữ nghĩa tương đồng giữa các từ đồng nghĩa ("movie" vs "film").
Dense Word2Vec (LAB 03 & 04)
  │
  ▼ Giới hạn: Vector tĩnh (Static Embedding), mỗi từ chỉ có một nghĩa cố định bất kể ngữ cảnh ("bank" trong bờ sông vs ngân hàng); phép cộng trung bình phá hủy hoàn toàn trật tự từ ("dog bites man" = "man bites dog").
LSTM / GRU (LAB 05)
  │
  ▼ Giới hạn: Ràng buộc tính toán tuần tự t-1 -> t ngăn cản song song hóa trên GPU; nút cổ chai thông tin (Information Bottleneck) tại trạng thái ẩn cuối h_T khiến văn bản dài bị suy hao tín hiệu.
Self-Attention (LAB 06)
  │
  ▼ Giới hạn: Hoàn toàn mất khái niệm về vị trí (Permutation Invariant) nếu không có cơ chế bù đắp; chỉ thực hiện tổ hợp tuyến tính thuần túy giữa các giá trị V mà thiếu khả năng biến đổi phi tuyến cục bộ; một đầu Attention duy nhất chỉ nhìn nhận được một kiểu tương quan.
Transformer Encoder (LAB 07)
  │
  ▼ Giới hạn còn tồn đọng:
    1. Chi phí tính toán và bộ nhớ tăng theo cấp số nhân bậc hai O(T^2), đặt giới hạn cứng về chiều dài ngữ cảnh trên GPU.
    2. Cần lượng dữ liệu huấn luyện khổng lồ để học các cấu trúc cú pháp từ đầu vì không có thiên kiến quy nạp (Inductive Bias) mạnh mẽ như mạng tích chập hay mạng hồi quy.
    3. Chưa có cơ chế học biểu diễn tự giám sát hai chiều quy mô lớn (Pre-trained Language Modeling) -> Đây chính là bước chuyển dịch sang BERT và GPT.
```

---

## 3. Bảng đối chuẩn và Kiểm chứng Giả thuyết (Prediction Verification)

| STT | Giả thuyết dự đoán (Prediction) | Dự đoán ban đầu | Kết quả thực nghiệm thực tế | Đánh giá | Giải thích cơ chế |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | Transformer xử lý song song hơn RNN? | Có, vượt trội hoàn toàn | Tốc độ huấn luyện nhanh hơn gần $45\%$ ($5.42s$ vs $9.85s$) | **ĐÚNG** | Toàn bộ chuỗi được nhân ma trận khối GEMM đồng thời trên GPU, không bị chặn bởi bước phụ thuộc thời gian $t-1 \to t$. |
| **2** | Bỏ Positional Encoding, Transformer có biết thứ tự không? | Không | Loss không đổi khi tráo ngẫu nhiên thứ tự các từ trong câu | **ĐÚNG** | Self-Attention chỉ tính tích vô hướng nội suy giữa các từ, không có hàm số nào phụ thuộc vào chỉ số vị trí nếu không có vector PE. |
| **3** | Phân biệt *"dog bites man"* vs *"man bites dog"* khi không có PE? | Không thể phân biệt | Vector biểu diễn sau Mean Pooling và Logits đầu ra hoàn toàn trùng khớp $100\%$ | **ĐÚNG** | Ma trận đầu ra chỉ là hoán vị hàng của nhau. Phép gộp trung bình làm triệt tiêu hoàn toàn sự khác biệt giữa hai hoán vị. |
| **4** | Hai Attention Heads có nhất thiết học cùng một mẫu hình? | Không | Head 1 tập trung vào từ kế cận, Head 2 tập trung cặp từ cảm xúc, Head 3 tập trung dấu chấm | **ĐÚNG** | Khởi tạo ma trận ngẫu nhiên và tối ưu hóa đa không gian con thúc đẩy các heads hội tụ vào các mẫu hình tương tác khác nhau. |
| **5** | Tăng số Attention Heads có chắc chắn làm tăng Accuracy? | Không chắc chắn | $h=4$ đạt Accuracy cao nhất ($87.33\%$), tăng lên $h=8$ giảm xuống $86.67\%$ | **ĐÚNG** | Khi $d_{model}=128$ cố định, $h=8$ khiến $d_k$ giảm xuống còn $16$ chiều, làm phân mảnh không gian biểu diễn và suy giảm độ sắc bén. |
| **6** | Transformer có luôn nhanh hơn RNN trên mọi sequence length? | Không | Khi $T=32, 64, 128$, Transformer nhanh hơn; khi $T \ge 1000$, chi phí $T^2$ bùng nổ | **ĐÚNG** | Sự bùng nổ số phần tử ma trận $T^2$ và lưu trữ activation VRAM triệt tiêu ưu thế song song khi chiều dài chuỗi quá lớn. |

---

## 4. Lời giải 9 câu hỏi kiểm tra năng lực cá nhân (Individual Learning Check)

### Câu 1: Tại sao Transformer cần Positional Encoding?
- **Trả lời:** Cơ chế Self-Attention là bất biến theo phép hoán vị (Permutation Invariant): nếu thay đổi thứ tự các từ trong câu, các điểm tích vô hướng $q_i^T k_j$ không đổi và tập hợp các vector ngữ cảnh đầu ra chỉ bị tráo đổi vị trí tương ứng. Do đó, nếu không có Positional Encoding, mô hình không thể phân biệt được câu chủ động và bị động (như *"dog bites man"* và *"man bites dog"*). Positional Encoding cộng trực tiếp một vector vị trí duy nhất vào vector embedding của từng từ để tiêm thông tin thứ tự thời gian vào mô hình.

### Câu 2: Multi-Head Attention khác một Attention Head đơn lẻ ở điểm nào?
- **Trả lời:** Một Attention Head đơn lẻ chỉ chiếu các token vào một không gian duy nhất và tính một phân phối trọng số trung bình. **Multi-Head Attention** chiếu các token thành $h$ không gian con độc lập thông qua các ma trận $W_i^Q, W_i^K, W_i^V$. Nhờ đó, tại cùng một bước tính toán, mô hình có thể đồng thời chú ý đến nhiều loại quan hệ khác nhau (ví dụ: một head chú ý quan hệ cú pháp chủ ngữ - vị ngữ, một head chú ý từ phủ định liền kề, một head chú ý quan hệ ngữ nghĩa tầm xa).

### Câu 3: Nếu $d_{model} = 128$ và có $8$ heads thì $d_k$ bằng bao nhiêu?
- **Trả lời:** Theo chuẩn kiến trúc Transformer, chiều không gian con của từng head được phân chia đều từ chiều mô hình tổng thể:
  $$d_k = \frac{d_{model}}{h} = \frac{128}{8} = 16$$
  Tổng chi phí tính toán của $8$ heads với $d_k = 16$ tương đương với chi phí của một head duy nhất với $d_{model} = 128$.

### Câu 4: Tại sao ma trận Attention có kích thước $T \times T$?
- **Trả lời:** Trong Self-Attention, mỗi token trong chuỗi $T$ tokens đóng vai trò là một Query, và nó cần phải tính điểm tương đồng với tất cả $T$ tokens trong cùng chuỗi đóng vai trò là Keys. Hàng thứ $i$ biểu diễn phân phối mức độ tập trung của token thứ $i$ lên toàn bộ $T$ tokens. Do đó, kích thước ma trận tương tác đầy đủ giữa tất cả các cặp token là $T \times T$.

### Câu 5: Residual Connection có vai trò gì trong Transformer Block?
- **Trả lời:** Residual Connection ($y = x + F(x)$) đóng hai vai trò sống còn:
  1. **Tạo đường cao tốc cho gradient (Gradient Highway):** Đạo hàm $\frac{\partial y}{\partial x} = I + \frac{\partial F(x)}{\partial x}$ luôn có số hạng đồng nhất $I$, giúp gradient truyền ngược trực tiếp qua hàng chục tầng mà không bị triệt tiêu về 0, giải quyết dứt điểm hiện tượng tiêu biến gradient.
  2. **Bảo tồn thông tin đầu vào:** Cho phép các tầng sâu chỉ cần học phần biến thiên dư (residual mapping $\Delta x$) thay vì phải tái tạo lại toàn bộ thông tin từ đầu.

### Câu 6: Feed-Forward Network (FFN) làm gì trong Transformer Block?
- **Trả lời:** Cơ chế Multi-Head Attention thực chất chỉ là một phép tổ hợp tuyến tính có trọng số giữa các vector giá trị $V$. **FFN** áp dụng hai lớp biến đổi tuyến tính kẹp giữa hàm kích hoạt phi tuyến (ReLU hoặc GELU) độc lập trên từng vị trí:
  $$FFN(x) = \max(0, x W_1 + b_1) W_2 + b_2$$
  Nhiệm vụ của FFN là biến đổi phi tuyến các đặc trưng ngữ cảnh vừa thu thập được, tăng cường năng lực biểu diễn và trừu tượng hóa tri thức tại từng vị trí.

### Câu 7: Tại sao Transformer có thể xử lý các token song song hơn RNN?
- **Trả lời:** Trong RNN, trạng thái ẩn tại thời điểm $t$ bắt buộc phải đợi trạng thái ẩn tại thời điểm $t-1$ hoàn thành ($\vec{h}_t = f(\vec{x}_t, \vec{h}_{t-1})$), tạo ra sự phụ thuộc tuần tự không thể phá vỡ. Trong Transformer, các biểu diễn $Q, K, V$ và ma trận tích vô hướng $Q K^T$ của toàn bộ $T$ tokens được tính đồng thời thông qua các phép nhân ma trận khối trên GPU, không có bất kỳ bước chờ đợi thời gian nào.

### Câu 8: Nếu chiều dài chuỗi tăng gấp đôi ($T \to 2T$), ma trận Attention thay đổi như thế nào?
- **Trả lời:** Kích thước ma trận Attention tăng từ $T \times T$ lên $(2T) \times (2T) = 4 T^2$. Tức là số lượng phần tử cần lưu trữ trong bộ nhớ và số phép tính tích vô hướng tăng **gấp 4 lần** (Tăng theo quy luật cấp số nhân bậc hai $\mathcal{O}(T^2)$).

### Câu 9: Vì sao không thể kết luận rằng Attention Weight cao đồng nghĩa với việc token đó là "lý do" mô hình đưa ra dự đoán?
- **Trả lời:** Vì:
  1. **Attention là tương quan số học bề mặt:** Trọng số Attention chỉ phản ánh độ lớn tích vô hướng trong không gian ẩn của lớp đó, không phải là lập luận nhân quả logic.
  2. **Các thành phần sau Attention làm thay đổi ngữ nghĩa:** Sau Attention còn có ma trận chiếu $W_O$, phép cộng Residual, phép chuẩn hóa LayerNorm, mạng phi tuyến FFN và bộ phân loại tuyến tính. Một token có Attention thấp vẫn có thể mang vector giá trị $V$ có độ lớn cực lớn và chi phối kết quả cuối cùng, hoặc ngược lại.

---

## 5. Tuyên bố sử dụng AI (AI Assistance Statement)

Căn cứ theo quy định tại **Mục 30. AI Policy**, bản tuyên bố dưới đây xác định rõ tính minh bạch và trung thực học thuật:

```yaml
AI Assistance Statement:
  Tool: Antigravity IDE (DeepMind Agentic Assistant)
  Purpose:
    - Hỗ trợ xây dựng khung mã nguồn script tạo đồ thị Matplotlib độ phân giải cao (generate_w7_plots.py).
    - Hỗ trợ định dạng cấu trúc bảng số liệu Markdown và kiểm tra tính hợp lệ của notebook JSON.
  Generated:
    - Khung hàm vẽ đồ thị nhiệt và đường cong học tập.
    - Cấu trúc mẫu của bảng đối chuẩn kết quả thực nghiệm.
  Modified:
    - Sinh viên loại bỏ toàn bộ các comment code trong mọi tệp mã nguồn python và notebook theo quy chuẩn nghiêm ngặt.
    - Tối ưu hóa cài đặt MultiHeadAttentionFromScratch để hỗ trợ Attention Mask và trích xuất trọng số phục vụ trực quan hóa.
  Verified:
    - Sinh viên trực tiếp thực hiện tính toán bằng tay từng bước giải tích tại calculations.md và đối chuẩn khớp số học tuyệt đối với code.
    - Sinh viên tự biện luận các giả thuyết khoa học tại prediction.md và phân tích định tính tại error_analysis.md.
    - Kiểm tra tính toàn vẹn 100% của toàn bộ 10 tệp tin deliverables.
```

### Năng lực giải thích các thành phần kiến trúc của sinh viên:
- **$Q, K, V$:** Các biểu diễn chiếu đại diện cho Truy vấn (Query), Khóa nhận diện (Key) và Nội dung thông tin (Value).
- **Multi-Head Attention:** Chiếu đồng thời thành nhiều không gian con để bắt trọn nhiều mối quan hệ tương tác khác nhau.
- **Positional Encoding:** Vector hàm sóng tuần hoàn đưa thông tin vị trí thời gian vào mô hình không phụ thuộc thứ tự.
- **Residual Connection ($x + F(x)$):** Tạo đường dẫn tắt bảo toàn thông tin đầu vào và giữ gradient dồi dào.
- **LayerNorm:** Chuẩn hóa trung bình và phương sai theo chiều đặc trưng để ổn định quá trình tối ưu hóa.
- **FFN:** Mạng truyền thẳng áp dụng phi tuyến độc lập tại từng vị trí để nâng cao năng lực biểu diễn ngữ nghĩa.

