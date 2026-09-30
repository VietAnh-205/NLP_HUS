## Câu 1. Nếu tăng $n$, mô hình nhận thêm thông tin gì?

Khi tăng $n$, mô hình sử dụng **nhiều từ trước đó hơn làm context** để dự đoán từ tiếp theo.

- Unigram: không sử dụng từ trước.
- Bigram: sử dụng 1 từ trước.
- Trigram: sử dụng 2 từ trước.

Ví dụ:

$$
P(w_i|w_{i-1})
$$

với Bigram và:

$$
P(w_i|w_{i-2},w_{i-1})
$$

với Trigram.

Vì vậy, tăng $n$ giúp mô hình có thêm thông tin về **ngữ cảnh cục bộ**.

---

## Câu 2. Tại sao tăng $n$ lại làm sparsity tăng?

Khi $n$ tăng, số lượng N-gram có thể có tăng rất nhanh.

Nếu vocabulary có $V$ từ thì số combination lý thuyết có thể lên tới:

$$
V^n
$$

Trong khi corpus chỉ chứa một phần rất nhỏ các N-gram đó.

Ví dụ, một Bigram có thể xuất hiện trong corpus nhưng Trigram chứa Bigram đó có thể chưa từng xuất hiện.

Do đó khi tăng $n$, ngày càng nhiều N-gram có **count bằng 0**, dẫn đến sparsity tăng.

---

## Câu 3. Tại sao smoothing cần thiết?

Trong Maximum Likelihood Estimation (MLE), nếu một N-gram chưa từng xuất hiện trong training corpus thì:

$$
P(w_i|context)=0
$$

Khi tính xác suất của cả sequence, chỉ cần một xác suất bằng 0 thì:

$$
P(W)=0
$$

và Perplexity trở thành:

$$
PPL=\infty
$$

Smoothing giải quyết vấn đề này bằng cách phân bổ một phần xác suất cho những N-gram chưa xuất hiện.

Ví dụ với Laplace smoothing:

$$
P(w_i|context)
=
\frac{Count(context,w_i)+1}
{Count(context)+V}
$$

Nhờ đó xác suất của unseen N-gram không còn bằng 0.

---

## Câu 4. Perplexity đo điều gì?

Perplexity đo mức độ **"bối rối" hoặc không chắc chắn của Language Model** khi dự đoán dữ liệu.

Công thức:

$$
PPL(W)
=
\exp
\left(
-\frac{1}{N}
\sum_{i=1}^{N}
\log P(w_i|context_i)
\right)
$$

Perplexity thấp thường có nghĩa model gán xác suất cao hơn cho dữ liệu thực tế.

Perplexity cao cho thấy model gán xác suất thấp hơn và "bối rối" hơn khi dự đoán dữ liệu.

Việc so sánh Perplexity chỉ có ý nghĩa khi các model được đánh giá trên **cùng dataset, cùng preprocessing và cùng cách tính**.

---

## Câu 5. Một model có perplexity thấp hơn có luôn tạo ra văn bản tốt hơn đối với con người không? Giải thích.

Không.

Perplexity chỉ đo khả năng của model trong việc **gán xác suất cho sequence**, chứ không trực tiếp đo chất lượng văn bản theo đánh giá của con người.

Một model có thể có Perplexity thấp nhưng văn bản sinh ra vẫn có thể:

- lặp lại từ hoặc câu;
- thiếu thông tin;
- thiếu logic;
- không phù hợp với ngữ cảnh;
- hoặc không tự nhiên đối với con người.

Ngoài ra, Perplexity còn phụ thuộc vào dataset và cách preprocessing. Vì vậy khi đánh giá chất lượng văn bản sinh ra, ngoài Perplexity còn cần các tiêu chí hoặc đánh giá khác.

---

## Câu 6. N-gram Language Model thất bại ở đâu khi so với cách con người hiểu ngôn ngữ?

N-gram Language Model chỉ nhìn thấy một **context có độ dài cố định**.

Ví dụ Trigram chỉ sử dụng hai từ trước đó:

$$
P(w_i|w_{i-2},w_{i-1})
$$

Do đó model khó nắm bắt được:

- quan hệ giữa các từ ở rất xa nhau;
- thông tin xuyên suốt một đoạn văn dài;
- ngữ cảnh dài hạn;
- ý nghĩa và kiến thức thế giới;
- quan hệ ngữ nghĩa phức tạp.

Con người có thể sử dụng thông tin từ nhiều câu hoặc nhiều đoạn trước đó để hiểu một câu hiện tại, trong khi N-gram truyền thống bị giới hạn bởi context length cố định.

Ngoài ra, khi tăng $n$ để lấy thêm context, số lượng N-gram tăng rất nhanh và gây ra **sparsity**.

---

## Câu 7. Nếu context dài 100 từ, Trigram có sử dụng được thông tin của 97 từ đầu không?

Không.

Trigram chỉ sử dụng **2 từ ngay trước từ đang được dự đoán**:

$$
P(w_i|w_{i-2},w_{i-1})
$$

Vì vậy nếu context có 100 từ:

```text
w1 w2 w3 ... w98 w99 w100