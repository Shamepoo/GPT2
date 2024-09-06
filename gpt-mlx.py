import random
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as o

batch_size = 32
block_size = 64
max_iter = 5000
eval_interval = 200
learning_rate = 1e-3
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# --------

mx.random.seed(1337)

# read it in to inspect it
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
# encoder: take a string, output a list of integers
def encode(s): return [stoi[c] for c in s]
# decoder: take a list of integers, output a string
def decode(l): return ''.join([itos[i] for i in l])


data = mx.array(encode(text))
# Let's now split up the data into train and validation sets
n = int(0.9*len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = mx.random.randint(0, len(data) - block_size, [batch_size])
    x = mx.stack([data[i.item():i.item()+block_size] for i in ix])
    y = mx.stack([data[i.item()+1:i.item()+block_size+1] for i in ix])
    return x, y


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        # (B, T, C)
        B, T, C = x.shape
        # (B, T, H)
        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)

        # (B, T, T)
        wei = Q@K.transpose(0, 2, 1)*(K.shape[-1]**(-0.5))
        tril = mx.tril(mx.ones((T, T)))
        # no mask fill in mlx, using where
        wei = mx.softmax(mx.where(tril, wei, -float('inf')), axis=-1)
        wei = self.dropout(wei)

        # (B, T, H)
        out = wei@V

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, n_head):
        super().__init__()
        self.heads = [Head(head_size) for _ in range(n_head)]
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        # (B, T, C)
        out = mx.concatenate([h(x) for h in self.heads], axis=-1)
        # (B, T, C)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def __call__(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa_heads = MultiHeadAttention(n_embd//n_head, n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def __call__(self, x):
        x = x+self.sa_heads(self.ln1(x))
        x = x+self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.postion_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def __call__(self, idx):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.postion_embedding_table(mx.arange(T))  # (T,C)

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        B, T = idx.shape
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # (B, T)
            # get the predictions
            logits = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = mx.softmax(logits, axis=-1)  # (B, C)
            # sample from the distribution
            # no multinomial in mlx, using random.choices
            idx_next = [random.choices(
                list(range(len(probs[i]))), probs[i].tolist(), k=1) for i in range(B)]

            idx_next = mx.array(idx_next, mx.int32).reshape((B, 1))  # (B, 1)
            # append sampled index to the running sequence
            idx = mx.concatenate((idx, idx_next), axis=1)  # (B, T+1)

        return idx


def loss_fn(model, input_data, expected):
    logits = model(input_data)
    loss = nn.losses.cross_entropy(logits, expected)
    return mx.mean(loss)


def estimate_loss(m):
    out = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            loss = loss_fn(m, xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    return out


def train(m: GPTLanguageModel):
    # Lazy Evaluation
    # https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html
    mx.eval(m.parameters())
    # mlx use this to do gradient descent
    vg = nn.value_and_grad(m, loss_fn)

    optimizer = o.AdamW(learning_rate)

    # increase number of steps for good results...
    for iter in range(max_iter):
        # sample a batch of data
        xb, yb = get_batch('train')
        # evaluate the loss
        loss, logits = vg(m, xb, yb)

        # update the model parameters
        optimizer.update(m, logits)
        # update the optimizer state
        mx.eval(m.parameters(), optimizer.state)

        if iter % eval_interval == 0 or iter == max_iter-1:
            losses = estimate_loss(m)

            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


m = GPTLanguageModel()
train(m)

print(decode(m.generate(idx=mx.ones((1, 1), mx.int32),
                        max_new_tokens=500)[0].tolist()))
