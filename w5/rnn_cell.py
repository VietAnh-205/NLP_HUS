import numpy as np

class SimpleRNNCell:
    def __init__(self, input_size, hidden_size, random_seed=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        limit_xh = 1.0 / np.sqrt(input_size)
        self.W_xh = np.random.uniform(-limit_xh, limit_xh, (hidden_size, input_size)).astype(np.float32)
        
        limit_hh = 1.0 / np.sqrt(hidden_size)
        self.W_hh = np.random.uniform(-limit_hh, limit_hh, (hidden_size, hidden_size)).astype(np.float32)
        
        self.b_h = np.zeros((hidden_size, 1), dtype=np.float32)

    def forward(self, x_t, h_prev):
        is_1d = (x_t.ndim == 1)
        if is_1d:
            x_t = x_t.reshape(-1, 1)
        if h_prev.ndim == 1:
            h_prev = h_prev.reshape(-1, 1)
        b = self.b_h.reshape(-1, 1) if self.b_h.ndim == 1 else self.b_h
        a_t = np.dot(self.W_xh, x_t) + np.dot(self.W_hh, h_prev) + b
        h_t = np.tanh(a_t)
        if is_1d:
            return h_t.flatten()
        return h_t

    def init_hidden(self):
        return np.zeros((self.hidden_size, 1), dtype=np.float32)

class SimpleRNNSequence:
    def __init__(self, input_size, hidden_size, random_seed=42):
        self.cell = SimpleRNNCell(input_size, hidden_size, random_seed=random_seed)
        self.hidden_size = hidden_size

    def forward(self, sequence_x, h_0=None):
        if h_0 is None:
            h_prev = self.cell.init_hidden()
        else:
            h_prev = h_0
            
        hidden_states = []
        for x_t in sequence_x:
            h_prev = self.cell.forward(x_t, h_prev)
            hidden_states.append(h_prev)
            
        return hidden_states, h_prev

class SimpleRNNClassifier:
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes=2, random_seed=42):
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        limit_emb = 1.0 / np.sqrt(embedding_dim)
        self.embeddings = np.random.uniform(-limit_emb, limit_emb, (vocab_size, embedding_dim)).astype(np.float32)
        
        self.rnn = SimpleRNNSequence(embedding_dim, hidden_size, random_seed=random_seed)
        
        limit_out = 1.0 / np.sqrt(hidden_size)
        self.W_hy = np.random.uniform(-limit_out, limit_out, (num_classes, hidden_size)).astype(np.float32)
        self.b_y = np.zeros((num_classes, 1), dtype=np.float32)

    def forward(self, token_ids):
        embedded_sequence = [self.embeddings[token_id] for token_id in token_ids]
        _, h_last = self.rnn.forward(embedded_sequence)
        logits = np.dot(self.W_hy, h_last) + self.b_y
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        return probabilities, h_last
