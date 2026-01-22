import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 
%matplotlib inline

words = open('names.txt', 'r').read().splitlines()
words[:8]

len(words)

# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
print(itos)

# build the dataset

block_size = 3 # context length: how many characters do we take to predict the next one?

X,Y = [], []

for w in words[:5]:
    
    print(w)
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        print(''.join(itos[i] for i in context), '----->', itos[ix])
        context = context[1:] + [ix] # crop and append

X = torch.tensor(X)
Y = torch.tensor(Y)

X.shape, X.dtype, Y.shape, Y.dtype

Embedding the 27 characters into a 2 dimensional space

C = torch.randn((27,2)) # embedding into a 2 dimensional space

C[5]

F.one_hot(torch.tensor(5),num_classes=27).float() @ C # same as C[5]

C[X].shape

X[13,2]

C[1]

emb = C[X]
emb.shape

W1 = torch.randn((6,100)) #W1 are weights #100 is the number of neurons in the hidden layer and 6 is the number of inputs to the hidden layer (3 chars * 2 dimensions)

b1 = torch.randn(100) # bias vector for hidden layer

# emb @ W1 + b1 # This cannot be done directly as emb and W1 have different shapes



                                                       
                                                       



