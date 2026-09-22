import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 

D1 = "cat eats fish"
D2 = "dog eats fish"
D3 = "cat likes fish"
vocab = ["cat", "dog", "eats", "fish", "likes"] 
x = np.array([1,1,1])
y = np.array([1, 1, 0]) 


#Bai 1

# cach 1 

docs = [D1, D2, D3]
result = [[int(word in doc.split()) for word in vocab] for doc in docs] 

# cach 2 

vectorier = CountVectorizer(vocabulary=vocab, binary=True) 
vector = vectorier.fit_transform(docs)
print(f"Vectorized output:\n{vector.toarray()}")


#bai 2 

len_d1 = len(D1.split()) 
tf_cat = len([word for word in D1.split() if word == 'cat']) / len_d1 
tf_eats = len([word for word in D1.split() if word == 'eats']) / len_d1 
tf_fish = len([word for word in D1.split() if word == 'fish']) / len_d1 
tf = {'cat': tf_cat, 
    'eats': tf_eats, 
    'fish': tf_fish
}

print(f"TF for cat: {tf['cat']:.3f}, TF for eats: {tf['eats']:.3f}, TF for fish: {tf['fish']:.3f}") 
print(f"Total TF: {sum(tf.values()):.0f}") 
 
#Bai 3 

df = [] 
for word in vocab: 
    count = 0
    for doc in docs: 
        if word in doc.split(): 
            count += 1 
    df.append(count) 

idf = {word: np.log(len(docs) / (count)) for word, count in zip(vocab, df)}
for i in range(len(vocab)): 
    print(f"IDF for {vocab[i]}: {idf[vocab[i]]:.3f}") 
print(f"Min IDF: {min(idf.values())}")

# fish tf-idf is 0 because it appears in all documents, making its IDF 0, and thus its TF-IDF is also 0.


#Bai 4 

tf_idf = {word: tf[word] * idf[word] for word in tf.keys()}
for word, value in tf_idf.items(): 
    print(f"TF_IDF for {word}: {value:.3f}")


#Bai 5 

cosine_similarities = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)) 
print(f"Cosine similarity between x and y: {cosine_similarities:.3f}")


#Bai 6 

D1 = "medical image classification"
D2 = "medical image analysis"
D3 = "natural language processing"

docs = [D1, D2, D3]

Q = "medical image classification"

"""
- D1 giống Q nhất nên cosine similarity giữa D1 và Q sẽ cao nhất, tiếp theo là D2 và cuối cùng là D3.
- "medical" và "image" xuất hiện nhiều nhất (2 lần) nên sẽ có idf thấp nhất 
- Ranking có thể không đổi, nhưng khoảng cách similarity giữa D1, D2, D3 sẽ thay đổi vì TF-IDF làm giảm trọng số của những từ phổ biến như medical và image.
"""
# code kiểm chứng : 

count_vectorizer = CountVectorizer()

X_count = count_vectorizer.fit_transform(docs)
Q_count = count_vectorizer.transform([Q])

similarity_count = cosine_similarity(Q_count, X_count)[0]

print("=== COUNT VECTOR ===\n")
for i, sim in enumerate(similarity_count): 
    print(f"Cosine similarity between D{i+1} and Q: {sim:.3f}")

tf_idf = TfidfVectorizer() 
X_tfidf = tf_idf.fit_transform(docs) 
Q_tfidf = tf_idf.transform([Q]) 
similarity_tfidf = cosine_similarity(Q_tfidf, X_tfidf)[0]

print("\n=== TF-IDF ===")

for i, sim in enumerate(similarity_tfidf):
    print(f"Q vs D{i+1}: {sim:.3f}")

