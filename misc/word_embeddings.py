#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.manual_seed(42)

#%%

word_to_ix = {"hello": 0, "world": 1}
vocab_size = len(word_to_ix)
embed_dim = 5

embeds = nn.Embedding(vocab_size, embed_dim)

lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)

hello_embed = embeds(lookup_tensor)
print(hello_embed)

#%%

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
# should tokenize

ngrams = [
    (
        [test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)],
        test_sentence[i]
    )
    for i in range(CONTEXT_SIZE, len(test_sentence))
]

print(ngrams[:3])

# %%
vocab = set(test_sentence)
word_to_ix = {word:i for i, word in enumerate(vocab)}

class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
    
#%%

losses = []
loss_fn = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

#%%
N_EPOCHS = 10

for epoch in range(N_EPOCHS):
    total_loss = 0
    for context, target in ngrams:
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
        # torch accumulates gradients, so before passing in a new instance
        # we zero out the gradients from the old instance
        model.zero_grad()
        # forward
        log_probs = model(context_idxs)
        # compute loss
        loss = loss_fn(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
        # backward, update gradients
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)

print(losses)

#%%
# get embeddings for a word
print(model.embeddings.weight[word_to_ix["beauty"]])

#%%
# CBOW

CONTEXT_SIZE = 2 # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(CONTEXT_SIZE, len(raw_text) - CONTEXT_SIZE):
    context = (
        [raw_text[i-j-1] for j in range(CONTEXT_SIZE)]
        + [raw_text[i+j+1] for j in range(CONTEXT_SIZE)]
    )
    target = raw_text[i]
    data.append((context, target))

print(data[:5])

#%%

"""
Notes:
Inputs will be of size [context_size*2].
"""

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        embeds = embeds.sum(axis=0)
        out = self.linear(embeds)
        log_probs = F.log_softmax(out, dim=0)
        return log_probs.view((1, -1))


#%%
def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

make_context_vector(data[0][0], word_to_ix)


#%%

losses = []
loss_fn = nn.NLLLoss()
model = CBOW(len(vocab), EMBEDDING_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.001)

N_EPOCHS = 100

for epoch in range(N_EPOCHS):
    total_loss = 0
    for context, target in data:
        context_vec = make_context_vector(context, word_to_ix)
        # torch accumulates gradients, so before passing in a new instance
        # we zero out the gradients from the old instance
        model.zero_grad()
        # forward
        log_probs = model(context_vec)
        # compute loss
        t = torch.tensor([word_to_ix[target]], dtype=torch.long)
        loss = loss_fn(log_probs, t)
        # backward, update gradients
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)

print(losses)

#%%
# get embeddings for a word
print(model.embeddings.weight[word_to_ix["are"]])
