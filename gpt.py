# read it in to inspect it
import torch
from torch.nn import functional as F
import torch.nn as nn
import torch  # we use PyTorch: https://pytorch.org

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


# let's now encode the entire text dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Let's now split up the data into train and validation sets
n = int(0.9*len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337)
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?
n_embd = 32

max_iters = 5000
eval_interval = 200
eval_iters = 200

device = torch.device("cpu")


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


class FeedForwardLayer(nn.Module):
    def __init__(self, n_embd) -> None:
        super().__init__()
        # 
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        self.q = nn.Linear(n_embd, head_size, bias=False)
        self.k = nn.Linear(n_embd, head_size, bias=False)
        self.v = nn.Linear(n_embd, head_size, bias=False)

    def forward(self, x):
        # [batch_size, time_step, n_embd]->[batch_size, time_step, head_size]
        x_q = self.q(x)
        x_k = self.k(x)
        x_v = self.v(x)
        # [batch_size, time_step, head_size] -> [batch_size, time_step, time_step]
        wei = x_q@x_k.transpose(-2, -1)
        # scal variance
        wei *= (x_q.shape[-1]**-0.5)

        # focus only on the previous information
        tril = torch.tril(wei)
        wei = F.softmax(torch.masked_fill(
            tril, tril == 0, float('-inf')), dim=-1)

        # [batch_size, time_step, time_step] -> # [batch_size, time_step, head_size]
        return wei@x_v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads*head_size, n_heads*head_size)

    def forward(self, x):
        # [batch_size, time_step, n_embd] -> [batch_size, time_step, head_size*n_heads]
        out = torch.cat([head(x) for head in self.heads], dim=-1)

        # [batch_size, time_step, head_size*n_heads] -> [batch_size, time_step, head_size*n_heads]
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, n_heads, head_size) -> None:
        super().__init__()
        self.m_heads = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForwardLayer(n_heads*head_size)

    def forward(self, x):
        # [batch_size, time_step, n_embd]
        x = x+self.m_heads(x)  # [batch_size, time_step, n_embd]
        x = x+self.ffwd(x)
        return x


class BigramLM(nn.Module):
    def __init__(self) -> None:
        super(BigramLM, self).__init__()
        n_heads = 4
        head_size = n_embd // n_heads
        self.tok_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        # [batch_size, time_step, n_embd]
        # self.m_heads = Head(head_size)
        self.block = nn.Sequential(
            *([Block(n_heads, head_size)]*3)
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, y=None):
        # x: [batch_size, time_step]
        # y: [batch_size, time_step]

        B, T = x.shape

        tok_emb = self.tok_embedding(x)  # [batch_size, time_step, n_embd]
        # [batch_size, time_step, n_embd]
        pos_emb = self.pos_embedding(
            torch.arange(T, device=device))

        x = tok_emb+pos_emb  # [batch_size, time_step, vocab_size]
        x = self.block(x)
        logits = self.lm_head(x)  # [batch_size, time_step, vocab_size]

        if y is not None:
            # logits = logits
            # logits ([batch_size, vocab_size, time_step])
            # y  [batch_size, vocab_size]
            loss = F.cross_entropy(logits.transpose(1, 2), y)
        else:
            loss = None

        return logits, loss

    def predict(self, x, max_new_tokens):
        # x : [batch_size, (1,2,3...)]
        # out: [batch_size, (1,2,3...), vocab_size]
        # pred: [1, 1]
        for _ in range(max_new_tokens):
            logits, _ = self(x[:, -block_size:])
            # focus on the last timestep (B, 1, C)
            logits = logits[:, -1, :]
            pred = F.softmax(logits, dim=-1)
            x_next = torch.multinomial(pred, num_samples=1)
            x = torch.cat([x, x_next], dim=1)
        return x


m = BigramLM().to(device)
xb, yb = get_batch('train')

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = m.forward(xb, yb)
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(m)
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


print(decode(m.predict(torch.zeros((1, 1), dtype=torch.long, device=device),
      max_new_tokens=100).tolist()[0]))


torch.arg