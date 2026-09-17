import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer 

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

#Bai 4 

tf_idf = {word: tf[word] * idf[word] for word in tf.keys()}
for word, value in tf_idf.items(): 
    print(f"TF_IDF for {word}: {value:.3f}")


#Bai 5 

cosine_similarity = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)) 
print(f"Cosine similarity between x and y: {cosine_similarity:.3f}")