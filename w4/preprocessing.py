import re
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

def clean_text(text, lowercase=True, remove_punct=False):
    if lowercase:
        text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    if remove_punct:
        text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def split_train_val_test(texts, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    np.random.seed(random_seed)
    n = len(texts)
    indices = np.random.permutation(n)
    
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    
    test_texts = [texts[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    
    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)

def train_word_embeddings_ppmi_svd(tokenized_corpus, embedding_dim=100, window_size=3, max_vocab=2000):
    word_counts = Counter()
    for doc in tokenized_corpus:
        word_counts.update(doc)
            
    vocab = [w for w, c in word_counts.most_common(max_vocab) if c >= 2]
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    v_size = len(vocab)
    
    if v_size == 0:
        return {}, embedding_dim
        
    pair_counts = Counter()
    for doc in tokenized_corpus:
        indices = [vocab_idx[w] for w in doc if w in vocab_idx]
        n_words = len(indices)
        for i in range(n_words):
            target = indices[i]
            start = max(0, i - window_size)
            end = min(n_words, i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    pair_counts[(target, indices[j])] += 1
                    
    if len(pair_counts) == 0:
        return {w: np.zeros(embedding_dim, dtype=np.float32) for w in vocab}, embedding_dim
        
    rows = np.array([p[0] for p in pair_counts.keys()], dtype=np.int32)
    cols = np.array([p[1] for p in pair_counts.keys()], dtype=np.int32)
    data = np.array(list(pair_counts.values()), dtype=np.float32)
    
    cooccur_csr = csr_matrix((data, (rows, cols)), shape=(v_size, v_size), dtype=np.float32)
    total_cooccur = float(data.sum())
    
    p_w = np.array(cooccur_csr.sum(axis=1)).flatten() / total_cooccur
    p_c = np.array(cooccur_csr.sum(axis=0)).flatten() / total_cooccur
    
    p_wc = data / total_cooccur
    expected = p_w[rows] * p_c[cols]
    
    pmi = np.log((p_wc + 1e-12) / (expected + 1e-12))
    ppmi_data = np.maximum(pmi, 0.0)
    
    ppmi_matrix = csr_matrix((ppmi_data, (rows, cols)), shape=(v_size, v_size), dtype=np.float32)
    
    n_comp = min(embedding_dim, v_size - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    embeddings = svd.fit_transform(ppmi_matrix)
    
    if n_comp < embedding_dim:
        pad = np.zeros((v_size, embedding_dim - n_comp), dtype=np.float32)
        embeddings = np.hstack([embeddings, pad])
        
    word_vectors = {w: embeddings[i] for i, w in enumerate(vocab)}
    return word_vectors, embedding_dim

def compute_document_embedding(tokens, word_vectors, embedding_dim):
    valid_vectors = [word_vectors[w] for w in tokens if w in word_vectors]
    if len(valid_vectors) == 0:
        return np.zeros(embedding_dim, dtype=np.float32)
    return np.mean(valid_vectors, axis=0).astype(np.float32)

def compute_corpus_embeddings(tokenized_corpus, word_vectors, embedding_dim):
    return np.array([
        compute_document_embedding(doc, word_vectors, embedding_dim)
        for doc in tokenized_corpus
    ], dtype=np.float32)

def compute_classification_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }

def compute_confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    
    return np.array([[tp, fn], [fp, tn]], dtype=int)
