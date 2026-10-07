import numpy as np

def build_vocabulary(corpus):
    vocab = set()
    for sentence in corpus:
        for word in sentence:
            vocab.add(word)
    return sorted(list(vocab))

def build_cooccurrence_matrix(corpus, vocabulary, window_size=1):
    vocab_index = {word: i for i, word in enumerate(vocabulary)}
    v_size = len(vocabulary)
    matrix = np.zeros((v_size, v_size), dtype=np.float32)
    for sentence in corpus:
        n = len(sentence)
        for i, target_word in enumerate(sentence):
            if target_word not in vocab_index:
                continue
            target_idx = vocab_index[target_word]
            start = max(0, i - window_size)
            end = min(n, i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    ctx_word = sentence[j]
                    if ctx_word in vocab_index:
                        ctx_idx = vocab_index[ctx_word]
                        matrix[target_idx, ctx_idx] += 1.0
    return matrix

def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

def most_similar(word, matrix, vocabulary, top_k=5):
    vocab_index = {w: i for i, w in enumerate(vocabulary)}
    if word not in vocab_index:
        return []
    target_idx = vocab_index[word]
    target_vec = matrix[target_idx]
    scores = []
    for w, idx in vocab_index.items():
        if w == word:
            continue
        sim = cosine_similarity(target_vec, matrix[idx])
        scores.append((w, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def word_similarity(word1, word2, matrix, vocabulary):
    vocab_index = {w: i for i, w in enumerate(vocabulary)}
    if word1 not in vocab_index or word2 not in vocab_index:
        return 0.0
    return cosine_similarity(matrix[vocab_index[word1]], matrix[vocab_index[word2]])

def compute_matrix_stats(matrix):
    total_elements = matrix.size
    non_zero = int(np.count_nonzero(matrix))
    sparsity = (1.0 - (non_zero / total_elements)) * 100.0 if total_elements > 0 else 0.0
    return {
        "shape": matrix.shape,
        "total_elements": total_elements,
        "non_zero": non_zero,
        "sparsity_percent": sparsity
    }
