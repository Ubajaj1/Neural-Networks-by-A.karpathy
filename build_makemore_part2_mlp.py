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

#Embedding the 27 characters into a 2 dimensional space

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

#Transform the 32*3*2 into 32*6 to perform the matrix multiplication with W1
                                                       
#We can concatenate the 3 characters together to get a 6 dimensional vector as input to the hidden layer
    

torch.cat([emb[:,0,:], emb[:,1,:], emb[:,2,:]],dim=1).shape  #this cannot be generalized lets say block size is different

# use unbind to generalize the concatenation of the characters in the context
# torch.unbind is used to split a tensor into a tuple of tensors along a given dimension                                      

torch.cat(torch.unbind(emb, dim=1),1).shape # this will split the tensor into 3 tensors of shape (32,2)

a = torch.arange(18)

a

a.view(9,2) # view is extremely efficient in tensor

a.storage() #this is how tensors are stored in memory as one dimensional vectors

h = torch.tanh(emb.view(-1,6)@W1 + b1) # this is the hidden layer output, when it is -1 it means that the number of rows is inferred from the number of columns and the total number of elements in the tensor

h.shape

#(emb.view(-1,6)@W1).shape #32,100

#b1.shape #1,100

# Addition will broadcast the above two terms

# 32,100
# 1,100

## creating the final layer
# input = 100, output = 27

W2 = torch.randn((100,27))
b2 = torch.randn(27)

logits = h @ W2 + b2 # this is the output of the network before applying softmax
logits.shape

counts = logits.exp() # this is equivalent to N in the softmax formula
prob = counts/ counts.sum(dim=1, keepdim=True) # this is equivalent to P in the softmax formula

prob.shape

prob[0].sum() # the probabilities for each example should sum to 1

torch.arange(32)

Y

prob[torch.arange(32), Y] # this will give us the probabilities assigned to the correct characters for each example

loss = -prob[torch.arange(32), Y].log().mean() # this is the negative log likelihood loss, we take the log of the probabilities and then take the mean over all examples


