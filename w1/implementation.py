from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
import numpy as np 


corpus = [
    "cat eats fish",  # D1
    "dog eats fish",  # D2
    "cat likes fish"  # D3
]
def build_vocabulary(docs): 
    vocab = set() 
    for doc in docs: 
        tokens = doc.split() 
        vocab.update(tokens) 
    return sorted(vocab) 

vocab = build_vocabulary(corpus) 
print(f"Vocabulary: {vocab}") 

def compute_counts(docs, vocab):
    counts = np.zeros((len(docs), len(vocab)))
    vocab_dict = {word: i for i, word in enumerate(vocab)}
    
    for i, doc in enumerate(docs):
        for word in doc.split():
            if word in vocab_dict:
                counts[i, vocab_dict[word]] += 1
                
    return counts
counts_matrix = compute_counts(corpus, vocab)
print("\nCount Matrix:\n", counts_matrix)

def compute_tf(counts):
    doc_lengths = counts.sum(axis=1, keepdims=True)
    tf_matrix = np.divide(counts, doc_lengths, out=np.zeros_like(counts), where=doc_lengths!=0)
    
    return tf_matrix

tf_matrix = compute_tf(counts_matrix)
print("\nTF Matrix:\n", tf_matrix)

def compute_idf(counts):
    N = counts.shape[0] 
    df = (counts > 0).sum(axis=0)
    idf_vector = np.log(N / df)
    return idf_vector

idf_vector = compute_idf(counts_matrix)
print("\nIDF Vector:\n", idf_vector)

def compute_tfidf(tf, idf):
    return tf * idf

tfidf_matrix = compute_tfidf(tf_matrix, idf_vector)
print("\nTF-IDF Matrix:\n", tfidf_matrix)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return dot_product / (norm1 * norm2)

sim_12 = cosine_similarity(counts_matrix[0], counts_matrix[1])
print(f"\nCosine similarity (Count vectors) D1 vs D2: {sim_12:.4f}")



sklearn_vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
sklearn_tfidf = sklearn_vectorizer.fit_transform(corpus).toarray()

print("\n--- SKLEARN TF-IDF MATRIX ---")
print(sklearn_tfidf)
print("Sklearn Vocabulary:", sklearn_vectorizer.get_feature_names_out())