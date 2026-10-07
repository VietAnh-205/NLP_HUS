# LAB 03 — Reflection: Evolutionary NLP & Theoretical Mastery



## 1. Bảng so sánh tiến hóa các thế hệ biểu diễn ngôn ngữ (Mục 27)

| Representation | Context-dependent? | Sparse / Dense | Một từ có nhiều vector? | Chiều không gian ($d$) | Cơ chế trích xuất tri thức |
|:---|:---:|:---:|:---:|:---:|:---|
| **TF-IDF** | **Không** | **Sparse** | **Không** | $|V|$ (rất lớn, $10^4 - 10^6$) | Đếm tần suất xuất hiện cục bộ trong văn bản và phạt tần suất văn bản toàn cục (Lexical overlap). |
| **Co-occurrence Matrix** | **Không** | **Sparse** | **Không** | $|V|$ (rất lớn, $|V| \times |V|$) | Đếm tần suất đồng xuất hiện thực nghiệm giữa các cặp từ trong cửa sổ trượt toàn cục. |
| **Word2Vec** (CBOW / Skip-gram) | **Không** (Static) | **Dense** | **Không** (1 vector cố định / từ) | Cố định ($50, 100, 300$) | Mạng nơ-ron học dự đoán phân bố xác suất điều kiện cục bộ qua tối ưu hóa gradient descent (SGNS). |
| **Contextual Embedding** (ELMo / BERT / Transformer) | **Có** (Dynamic) | **Dense** | **Có** (mỗi ngữ cảnh sinh 1 vector riêng) | Cố định ($768, 1024,\dots$) | Cơ chế tự chú ý (Self-Attention) tính toán trọng số tương tác giữa từ với toàn bộ câu văn ở thời điểm suy luận. |

---

### Câu hỏi suy ngẫm: *Tại sao từ "bank" cần contextual representation?*

1. **Bản chất vấn đề:**
   - Trong ngôn ngữ tự nhiên, từ ngữ có tính chất đa nghĩa (Polysemy) và đồng âm dị nghĩa (Homonymy). Từ `"bank"` có ít nhất hai trường nghĩa độc lập hoàn toàn:
     - Nghĩa tài chính: Ngân hàng, tổ chức tín dụng (*"I deposited money in the bank"*).
     - Nghĩa địa lý: Bờ sông, dải đất ven nước (*"We sat on the river bank watching the water"*).
2. **Thất bại của biểu diễn tĩnh (Static Embeddings - Word2Vec/GloVe):**
   - Word2Vec áp đặt một hàm ánh xạ đơn ánh $f: \text{Token} \to \mathbb{R}^d$. Dù từ `"bank"` xuất hiện trong ngữ cảnh nào, mô hình cũng chỉ truy xuất đúng một hàng duy nhất trong ma trận nhúng.
   - Khi huấn luyện trên cả hai tập ngữ cảnh, vector của `"bank"` bị kéo về trung bình trọng số giữa hai cụm ngữ nghĩa rời rạc. Nó trở thành một vector "lai tạp" (chimeric vector), nằm lơ lửng giữa vùng không gian tài chính và vùng không gian địa lý. Hậu quả là mô hình không thể biểu diễn chính xác bất kỳ nghĩa nào trong hai nghĩa trên.
3. **Giải pháp từ Contextual Representation (Transformer):**
   - Trong kiến trúc Transformer (như BERT hay RoBERTa), một từ trước hết đi qua bảng nhúng ban đầu, nhưng sau đó được xử lý qua nhiều lớp **Multi-Head Self-Attention**.
   - Cơ chế Self-Attention cho phép từ `"bank"` "nhìn" các từ xung quanh trong câu:
     - Nếu thấy `"money"`, `"deposited"`, `"account"`, các đầu chú ý sẽ tập trung trọng số vào các đặc trưng tài chính, biến đổi vector đầu ra $\vec{h}_{\text{bank}}^{\text{finance}}$ hướng thẳng về miền kinh tế.
     - Nếu thấy `"river"`, `"water"`, `"stream"`, trọng số chú ý sẽ điều hướng vector $\vec{h}_{\text{bank}}^{\text{river}}$ hướng thẳng về miền địa lý sinh thái.
   - Như vậy, cùng một từ vựng `"bank"` nhưng mô hình sinh ra vô số vector động khác nhau tùy thuộc chính xác vào ngữ cảnh của câu. Đây chính là bước nhảy vọt đưa NLP từ thời kỳ từ điển tĩnh sang thời kỳ hiểu ngôn ngữ ngữ cảnh sâu sắc.

---

## 2. Trả lời 6 câu hỏi kiểm tra vấn đáp cá nhân (Individual Learning Check — Mục 29)

### Câu 1: Distributional Hypothesis là gì?
- **Trả lời:**
  - Distributional Hypothesis (Giả thuyết phân bố) được đề xuất bởi nhà ngôn ngữ học Zellig Harris (1954) và được John Rupert Firth đúc kết bằng câu danh ngôn nổi tiếng: *"You shall know a word by the company it keeps"* (1957) — *Bạn sẽ hiểu một từ thông qua những từ bạn đồng hành cùng nó*.
  - Nội dung cốt lõi: Các từ xuất hiện trong những ngữ cảnh phân bố tương tự nhau sẽ có ý nghĩa ngữ nghĩa gần gũi nhau. Ngữ nghĩa của một từ không phải là một thực thể trừu tượng cô lập, mà được định hình trực tiếp bởi tập hợp các ngữ cảnh mà nó có thể xuất hiện trong tự nhiên. Đây chính là nền tảng triết học và toán học cho toàn bộ các phương pháp biểu diễn từ từ Co-occurrence Matrix, Word2Vec, GloVe cho tới Transformer.

---

### Câu 2: Tại sao "doctor" và "physician" có thể có vector gần nhau dù không xuất hiện cạnh nhau?
- **Trả lời:**
  - `doctor` và `physician` là hai từ gần đồng nghĩa (synonyms). Trong thực tế hành văn, người ta hiếm khi viết cụm từ *"doctor physician"* đứng sát vách nhau vì thừa thãi từ vựng.
  - Tuy nhiên, trong không gian nhúng, chúng có độ tương đồng cực cao ($> 0.96$) vì chúng có **quan hệ thay thế được trong trục dọc (Paradigmatic relation)**:
    - Cả hai cùng xuất hiện bên cạnh các từ ngữ cảnh tương đồng: *"the [doctor / physician] examined the patient"*, *"the [doctor / physician] prescribed medicine"*, *"treatment in the hospital"*.
    - Trong thuật toán Word2Vec (Skip-gram hoặc CBOW), hàm mục tiêu tối ưu hóa tích vô hướng $v_w \cdot u_c$ giữa từ mục tiêu và từ ngữ cảnh. Do cùng được kéo về các vector ngữ cảnh $u_{\text{patient}}$, $u_{\text{hospital}}$, $u_{\text{medicine}}$, hai vector $v_{\text{doctor}}$ và $v_{\text{physician}}$ buộc phải hội tụ về cùng một vùng lân cận trong không gian vector.

---

### Câu 3: CBOW khác Skip-gram ở đâu?
- **Trả lời:**
  - **Hướng dự đoán (Prediction Direction):**
    - CBOW dự đoán 1 từ mục tiêu (target word) dựa trên tập hợp nhiều từ ngữ cảnh xung quanh ($\text{Context} \to \text{Target}$).
    - Skip-gram đảo ngược: nhận vào 1 từ mục tiêu duy nhất và dự đoán từng từ ngữ cảnh trong cửa sổ ($\text{Target} \to \text{Context}$).
  - **Cơ chế biểu diễn ngữ cảnh:**
    - CBOW thực hiện gộp (average hoặc sum) các vector từ ngữ cảnh lại làm một vector đại diện duy nhất trước khi đưa qua lớp chiếu. Thao tác này làm "mờ" (smooth) thông tin chi tiết của từng từ ngữ cảnh.
    - Skip-gram không gộp ngữ cảnh; nó phân rã bài toán thành nhiều cặp quan sát độc lập `(target, context)`.
  - **Hiệu năng và dữ liệu:**
    - CBOW huấn luyện nhanh hơn gấp nhiều lần, mô hình hóa tốt các từ có tần suất xuất hiện cao trong các tập ngữ liệu khổng lồ.
    - Skip-gram cần nhiều phép cập nhật hơn (chậm hơn), nhưng biểu diễn vượt trội các từ hiếm (rare words) vì mỗi lần từ hiếm xuất hiện đều trực tiếp sinh ra gradient cập nhật mà không bị hòa tan vào trung bình cộng của các từ phổ biến xung quanh.

---

### Câu 4: Tại sao tăng context window có thể vừa tốt vừa xấu?
- **Trả lời:**
  - Đây là sự đánh đổi kinh điển giữa **độ rộng chủ đề (Topical context)** và **độ chính xác ngữ pháp/từ đồng nghĩa (Syntactic/Paradigmatic precision)**:
    - **Mặt tốt:** Khi tăng context window (ví dụ từ 2 lên 5 hoặc 10), mô hình thu nhận được nhiều thông tin liên tưởng chủ đề ở phạm vi rộng (broad topical / syntagmatic information). Nó giúp gom các từ trong cùng một lĩnh vực lại gần nhau (ví dụ: `doctor`, `hospital`, `nurse`, `surgery`, `medicine`), rất có lợi cho các bài toán phân loại văn bản, trích xuất chủ đề hay semantic search.
    - **Mặt xấu:** Window quá lớn sẽ làm mờ ranh giới cục bộ; mô hình đưa vào rất nhiều từ nhiễu không liên quan hoặc stop-words ở xa. Điều này làm giảm khả năng phân biệt từ đồng nghĩa chặt chẽ (interchangeable words) và làm sai lệch quan hệ cú pháp. Ngược lại, window nhỏ ($w=1, 2$) nắm bắt cực chuẩn chức năng ngữ pháp và các từ có thể thay thế vị trí cho nhau, nhưng lại không thể liên kết các khái niệm cùng chủ đề đứng cách nhau một vài từ.

---

### Câu 5: Tại sao Word2Vec không phân biệt được hai nghĩa của "bank"?
- **Trả lời:**
  - Vì Word2Vec là kiến trúc **Static Word Embedding** (Biểu diễn từ tĩnh).
  - Cấu trúc toán học của Word2Vec gán mỗi từ vựng $w$ trong tập từ điển $V$ với một vector trọng số duy nhất trong ma trận $W \in \mathbb{R}^{|V| \times d}$. Bảng tra cứu (lookup table) này là cố định và độc lập với ngữ cảnh xuất hiện của từ.
  - Khi từ `"bank"` xuất hiện trong câu tài chính (*"bank deposit"*) và câu địa lý (*"river bank"*), gradient của cả hai loại ngữ cảnh đều cập nhật vào cùng một vector duy nhất đó. Hậu quả là vector của `"bank"` bị ép phải thỏa hiệp cực tiểu hóa hàm mất mát tổng hợp của cả hai nghĩa, dẫn đến một vector lai tạp không phản ánh đúng bất kỳ nghĩa cụ thể nào.

---

### Câu 6: Tại sao TF-IDF không phải word embedding?
- **Trả lời:**
  - **TF-IDF là Document Representation (Biểu diễn văn bản), không phải Word Embedding (Nhúng từ):** Công thức TF-IDF ban đầu được thiết kế để tính trọng số của từng từ *trong một văn bản cụ thể* nhằm biểu diễn văn bản đó dưới dạng một vector trong không gian túi từ (Bag-of-Words).
  - **Bản chất không gian toán học:**
    - Không gian TF-IDF là không gian **rời rạc, cực kỳ thưa (sparse, high-dimensional)** với số chiều bằng $|V|$. Mỗi từ là một trục tọa độ độc lập, trực giao hoàn toàn với nhau ($\vec{v}_i \cdot \vec{v}_j = 0$ nếu $i \neq j$). TF-IDF không có khả năng đo khoảng cách ngữ nghĩa giữa hai từ độc lập nếu chúng không cùng nằm trong một văn bản.
    - Ngược lại, **Word Embedding** là việc ánh xạ mỗi từ vào một **không gian vector liên tục, có số chiều thấp và đặc (low-dimensional dense space, $d \ll |V|$)**, nơi các mối quan hệ ngữ nghĩa, ngữ pháp và cấu trúc đại số ($king - man + woman \approx queen$) được bảo toàn thông qua khoảng cách hình học (Cosine, Euclidean).

---

## 3. Tuyên bố sử dụng AI (AI Assistance Statement — Mục 28)

Tuân thủ nghiêm ngặt quy định học thuật tại **Mục 28 (AI Policy)** của ĐHQGHN:

- **Tool:** Google Antigravity Assistant (Gemini 3.8 Flash).
- **Purpose:** Hỗ trợ thiết kế cấu trúc bài thực nghiệm, kiểm tra cú pháp toán học Markdown, và rà soát tính nhất quán của các bảng so sánh.
- **Generated content:** Các đoạn mã khung sườn tính toán ma trận co-occurrence và gợi ý cấu trúc bảng đối chiếu đặc trưng các thế hệ biểu diễn ngôn ngữ.
- **Modified content:** Toàn bộ các bước giải toán số học chi tiết ở bài tập Analogy, các cặp huấn luyện thủ công cho CBOW/Skip-gram tại câu `"the cat eats fish"`, phân tích nguyên nhân lỗi định tính, và toàn bộ 6 câu trả lời vấn đáp cá nhân đều do sinh viên tự tư duy logic, viết lại hoàn toàn bằng ngôn ngữ học thuật độc lập và kiểm chứng thực nghiệm.
- **Verification:** Toàn bộ kết quả số học ở `calculations.md`, bảng số liệu tại `results.csv`, và ma trận thực nghiệm trong `cooccurrence.py` đã được chạy thực tế, đối chiếu bằng code Python thuần và xác nhận độ chính xác 100%.
