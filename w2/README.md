# LAB 02 — N-gram Language Models

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng | Học kỳ I - 2026

**Giảng viên / TA:** Phạm Ngọc Hải

## 📖 1. Tổng quan Lab

Buổi lab này hướng tới việc chuyển hóa kiến thức lý thuyết về **N-gram Language Models**, **Maximum Likelihood Estimation (MLE)**, kỹ thuật làm trơn **Laplace Smoothing**, và độ đo đánh giá **Perplexity (PPL)** thành kỹ năng thực hành. Bạn sẽ đi qua một chu trình nghiên cứu hoàn chỉnh:

> Tính toán tay → Đưa ra dự đoán → Code thực nghiệm → Xây dựng ứng dụng (Next-word Prediction & Sentence Ranking) → Đánh giá Perplexity → Phân tích lỗi (Error Analysis).

**Mục tiêu cốt lõi:** 
- Nắm vững nguyên lý ước lượng xác suất chuỗi từ và câu bằng mô hình Markov $n$-gram ($n=1, 2, 3$).
- Hiểu rõ vấn đề dữ liệu thưa thớt (Data Sparsity), hiện tượng Zero-frequency và vai trò của kỹ thuật Smoothing.
- Ứng dụng Language Model vào bài toán dự đoán từ tiếp theo (Next-word Prediction) và xếp hạng câu (Sentence Ranking).
- Phân tích giới hạn của context length trong n-gram truyền thống và lý giải bước chuyển dịch sang các mô hình ngôn ngữ nơ-ron (Neural LM, Word Embeddings, RNN, Transformer).

## 📁 2. Cấu trúc thư mục nộp bài (Deliverables)

```
lab02/
├── README.md
├── calculations.md
├── prediction.md
├── ngram_lm.py
├── experiments.ipynb
├── results.csv
├── error_analysis.md
└── reflection.md
```

Trong đó:

| File | Nội dung |
|:---|:---|
| `calculations.md` | Tính toán bằng tay |
| `prediction.md` | Dự đoán trước experiment |
| `ngram_lm.py` | Implementation |
| `experiments.ipynb` | Experiments |
| `results.csv` | Perplexity + prediction results |
| `error_analysis.md` | Phân tích lỗi |
| `reflection.md` | Câu hỏi tổng kết |

