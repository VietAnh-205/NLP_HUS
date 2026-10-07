# LAB 03 — Error Analysis & Polysemy Breakdown


## 1. Bảng tổng hợp các trường hợp quan sát

| Phân loại | Cặp từ vựng / Truy vấn | Độ tương đồng quan sát (Observed Sim) | Kỳ vọng ngữ nghĩa (Expected Relation) | Trạng thái đánh giá |
|:---|:---|:---:|:---|:---:|
| **Correct 1** | `doctor` — `nurse` | $0.9669$ | Đồng vai trò chuyên môn y tế (Medical professions) | Hợp lý (High Semantic Fit) |
| **Correct 2** | `king` — `queen` | $0.9912$ | Cặp tương đồng loại suy vương quyền (Royal analogy pair) | Chuẩn xác (Strong Royalty Match) |
| **Correct 3** | `cat` — `dog` | $0.9616$ | Động vật nuôi trong nhà (Domestic pets) | Chuẩn xác (Same Taxonomy Class) |
| **Surprising 1** | `computer` — `banana` (Window = 2) | $0.9901$ | Trực giao / Không liên quan ($\approx 0.0$ hoặc rất thấp) | Sai lệch nghiêm trọng (Spurious Similarity) |
| **Surprising 2** | `doctor` — `disease` | $0.9483$ | Đối tượng can thiệp vs Tác nhân bệnh lý (Actor vs Condition) | Trộn lẫn quan hệ (Syntagmatic Conflation) |
| **Surprising 3** | `bank` — `river` & `bank` — `money` | $0.9792$ & $0.8502$ | Hai nghĩa tách biệt hoàn toàn (Polysemy: Finance vs Nature) | Sụp đổ không gian (Polysemy Collapse) |

---

## 2. Phân tích chi tiết 3 trường hợp đúng (Correct Similarities)

### 2.1. Cặp 1: `doctor` $\longrightarrow$ `nurse`
- **Observed:** Cosine similarity đạt $0.9669$ (Top-1 trong danh sách các từ gần nhất với `doctor`).
- **Expected:** Kỳ vọng hai từ có độ tương đồng rất cao vì cùng là các chuyên gia y tế, nhân viên chăm sóc sức khỏe bệnh nhân trong bệnh viện.
- **Possible explanation:**
  - Cả hai từ cùng chia sẻ tập từ ngữ cảnh kế cận gần như đồng nhất: thường xuyên xuất hiện với các vị ngữ như *treat, care, help, examine* và các danh từ *patient, hospital, clinic*.
  - Theo Distributional Hypothesis (Harris, 1954), hai từ có phân bố ngữ cảnh giống nhau sẽ có vector nhúng rất gần nhau trong không gian vector.
- **Evidence from corpus:**
  - `"a doctor and nurse provide medical treatment in the clinical center"`
  - `"the nurse cares for the patient in the hospital ward"`
  - `"the doctor examines the sick patient in the hospital"`
  $\implies$ Sự hoán đổi vị trí trực tiếp trong các câu chứa `patient` và `hospital` tạo ra lực kéo gradient mạnh mẽ đưa hai vector về gần nhau.

---

### 2.2. Cặp 2: `king` $\longrightarrow$ `queen`
- **Observed:** Cosine similarity đạt $0.9912$ ở Window 5 và $0.9965$ ở Window 10.
- **Expected:** Kỳ vọng độ tương đồng cực cao do thuộc cùng tầng lớp vương triều, đại diện cho cặp đối xứng giới tính chuẩn mực nhất trong ngôn ngữ.
- **Possible explanation:**
  - Trong ngữ liệu, hai từ này luôn xuất hiện song hành hoặc trong các cấu trúc đối xứng (*"king rules with queen"*, *"king and queen sat upon the royal throne"*, *"the man became king while the woman became queen"*).
  - Tần suất đồng xuất hiện (co-occurrence) trong cùng một cửa sổ context cực kỳ đậm đặc, khiến hàm mất mát Skip-gram liên tục cập nhật hướng vector của chúng về cùng một cụm cực nhỏ (tight cluster).
- **Evidence from corpus:**
  - `"the king rules the royal kingdom with the queen"`
  - `"the king and queen sat upon the royal throne"`
  - `"the prince is the son of the king and queen"`

---

### 2.3. Cặp 3: `cat` $\longrightarrow$ `dog`
- **Observed:** Cosine similarity đạt $0.9616$ (Window 5) và $0.9896$ (Window 10).
- **Expected:** Kỳ vọng hai từ có độ tương đồng cao vì cùng là thú cưng 4 chân, vật nuôi phổ biến trong gia đình.
- **Possible explanation:**
  - Cả hai cùng xuất hiện bên cạnh các động từ hoạt động thú cưng (*play, eat, sleep, chase, run*) và danh từ phân loại (*pet, puppy, kitten*).
  - Mô hình học được quan hệ phân loại đồng cấp (co-hyponyms) thông qua việc chia sẻ các thuộc tính ngữ cảnh sinh học và môi trường sống.
- **Evidence from corpus:**
  - `"the cat plays with the small dog in the garden"`
  - `"the dog and cat are friendly domestic pets"`
  - `"a puppy is a young dog and a kitten is a young cat"`

---

## 3. Phân tích chi tiết 3 trường hợp sai / bất ngờ (Incorrect / Surprising Similarities)

### 3.1. Sai lệch 1: `computer` — `banana` tại Window = 2 ($0.9901$)
- **Observed:** Tại kích thước cửa sổ rất nhỏ ($w = 2$), độ tương đồng giữa `computer` (công nghệ) và `banana` (hoa quả) vọt lên mức $0.9901$, gần như tương đương với hai từ đồng nghĩa.
- **Expected:** Kỳ vọng độ tương đồng xấp xỉ $0.0$ hoặc giá trị âm nhỏ vì chúng hoàn toàn không có liên hệ ngữ nghĩa hay chủ đề.
- **Possible explanation:**
  - **Corpus nhỏ & Cấu trúc ngữ pháp trùng lặp (Syntactic Template Sharing):** Khi tập dữ liệu nhỏ, cả hai từ đều xuất hiện trong các mẫu câu ngữ pháp đơn giản có cấu trúc tương tự nhau: `[the] + [NOUN] + [VERBs] + [OBJECT]`.
  - **Context Window quá hẹp ($w=2$):** Với $w=2$, mô hình bị "mù chủ đề" (blind to global topic). Cả `computer` và `banana` đều đứng cạnh các stop-words như `"the"`, `"is"`, `"and"`. Do mạo từ `"the"` xuất hiện dày đặc quanh cả hai từ, vector của chúng bị kéo về hướng của mạo từ `"the"`.
  - **Hiện tượng giải phóng khi tăng Window:** Khi tăng lên Window 5 ($0.6518$) và Window 10 ($0.4247$), cửa sổ mở rộng bao trùm các từ chủ đề đặc thù (*software, algorithm, data* vs *fruit, sweet, yellow, eat*), giúp mô hình tách biệt rõ rệt hai miền ngữ nghĩa.
- **Evidence from corpus:**
  - `"the computer executes software program"` (bên trái `computer` là `the`)
  - `"the sweet banana is a healthy fruit"` (cạnh `banana` có `sweet` và `is`)
  - Việc thiếu hụt dữ liệu âm tính (negative sampling) giữa hai từ trong cửa sổ hẹp dẫn đến việc vector chưa đủ bước phân rã không gian.

---

### 3.2. Sai lệch 2: `doctor` $\longrightarrow$ `disease` ($0.9483$)
- **Observed:** `disease` xuất hiện trong Top-5 từ gần nhất với `doctor` với độ tương đồng rất cao ($0.9483$).
- **Expected:** Bác sĩ (`doctor`) là chủ thể điều trị (agent/healer), còn bệnh tật (`disease`) là trạng thái bệnh lý cần loại bỏ (condition/target). Về mặt quan hệ ngữ nghĩa (taxonomic relation), chúng không phải là từ đồng nghĩa, thậm chí mang tính chất đối nghịch về mặt tác nhân.
- **Possible explanation:**
  - **Sự trộn lẫn giữa Quan hệ Cùng Ngữ Cảnh (Syntagmatic) và Quan hệ Thay Thế Được (Paradigmatic):**
    - Word2Vec đo lường sự tương đồng phân bố (distributional similarity), không đo lường sự tương đồng thực thể logic.
    - Bác sĩ và bệnh tật có quan hệ liên tưởng chủ đề cực kỳ chặt chẽ (topical association / syntagmatic relation). Trong mọi văn bản nói về y tế, hễ có bác sĩ xuất hiện thì tỷ lệ bệnh tật xuất hiện là rất lớn.
    - Word2Vec không phân biệt được: "X điều trị Y" khác với "X là một loại Y". Cả hai đều biến thành tích vô hướng vector cao do liên tục xuất hiện cùng nhau.
- **Evidence from corpus:**
  - `"the doctor diagnosed the disease and admitted patient to hospital"`
  - `"the patient received clinical therapy for the chronic disease"`
  - `"effective treatment cured the rare disease in the clinic"`

---

### 3.3. Sai lệch 3: Hiện tượng sụp đổ không gian do từ đa nghĩa — `bank`
- **Observed:**
  - $\text{sim}(\text{bank}, \text{money}) = 0.8502$
  - $\text{sim}(\text{bank}, \text{river}) = 0.9792$
  - $\text{sim}(\text{bank}, \text{water}) = 0.9790$
  - Trong khi đó: $\text{sim}(\text{money}, \text{river}) = 0.7837$
- **Expected:**
  - Từ `bank` trong ngữ cảnh tài chính phải gần `money`, `deposit`, `financial` và cách xa `river`.
  - Từ `bank` trong ngữ cảnh bờ sông phải gần `river`, `water`, `stream` và cách xa `money`.
- **Possible explanation (Phân tích chuyên sâu mục 26):**
  - **Giới hạn cố hữu của Static Word Embedding:** Mỗi từ $w$ chỉ có duy nhất **một** vector $\vec{v}_w$ trong ma trận trọng số $W$.
  - Trong quá trình huấn luyện bằng SGD/SGNS:
    - Khi gặp câu tài chính (*"deposited money in the bank"*), gradient kéo $\vec{v}_{\text{bank}}$ về hướng của $\vec{v}_{\text{money}}$.
    - Khi gặp câu địa lý (*"sat on the river bank watching water"*), gradient lại kéo $\vec{v}_{\text{bank}}$ về hướng của $\vec{v}_{\text{river}}$ và $\vec{v}_{\text{water}}$.
  - Kết quả là vector $\vec{v}_{\text{bank}}$ trở thành một **vector trung bình thỏa hiệp (weighted average centroid)** nằm lơ lửng ở giữa hai cụm ngữ nghĩa hoàn toàn không liên quan.
  - Hậu quả: Một người dùng tìm kiếm văn bản kinh tế tài chính có thể bị gợi ý văn bản về du lịch sông ngòi do vector `bank` vô tình kéo hai chủ đề này lại gần nhau!
- **Evidence from corpus:**
  - Nghĩa 1 (Finance): `"the customer deposited money into the bank account"`, `"the bank provides financial loans and saves paper money"`.
  - Nghĩa 2 (Geography): `"we sat on the river bank watching the water stream"`, `"clear water flows along the peaceful river bank"`.

---

## 4. Tổng kết các nguyên nhân cốt lõi gây sai số trong Word Embeddings

1. **Kích thước Corpus (Small Corpus):** Số lượng ngữ cảnh quan sát được quá ít, khiến mô hình nhạy cảm với các mẫu xuất hiện ngẫu nhiên.
2. **Thiên kiến miền dữ liệu (Domain Bias):** Nếu corpus chỉ toàn văn bản y tế, mọi từ vựng đều bị "nhiễm" ngữ cảnh lâm sàng.
3. **Cửa sổ ngữ cảnh (Context Window Size):** Window quá nhỏ bắt nhầm tương đồng cú pháp bề mặt; window quá lớn gom cả từ rác và kéo các chủ đề xa lạ lại gần nhau.
4. **Hiện tượng từ đa nghĩa (Polysemy):** Không thể biểu diễn nhiều nghĩa bằng 1 vector tĩnh duy nhất (Single Static Vector limitation).
5. **Đồng xuất hiện chủ đề (Topical Conflation):** Nhầm lẫn giữa từ đồng nghĩa (synonyms) và từ liên quan chủ đề (topically related / antonyms / co-occurring words).
