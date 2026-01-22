import torch
import matplotlib.pyplot as plt
%matplotlib inline
import torch.nn.functional as F

words = open('names.txt','r').read().splitlines()

words[:10]

len(words)

min(len(w) for w in words)

max(len(w) for w in words)

b = {}
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1,ch2 in zip(chs,chs[1:]):
        bigram = (ch1,ch2)
        b[bigram] = b.get(bigram,0) + 1
        #print(ch1,ch2)

sorted(b.items(),key = lambda kv: -kv[1])

N = torch.zeros((27,27), dtype = torch.int32)


chars = sorted(list(set(''.join(words))))

# mapping from character to index
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.']= 0
stoi

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1,ix2]+=1

itos = {i:s for s,i in stoi.items()}


plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off');

N[0]

p= N[0].float()
p = p /p.sum()
p

g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p,num_samples=1, replacement=True, generator=g).item()
ix
itos[ix]

P = N.float()

# can we divide P which is 27,27 matrix by a 27,1 matrix?
# Answer is Broadcast Join in Tensors, What is broadcast join?
# Two Tensors are broadcastable if 
# Each Tensor has at least one dimension
# When iterating over the dimension sizes, starting at the trailing dimension, the dimension sizes must either be equal, one of them is 1, or one of them does not exist

P /= P.sum(dim=1,keepdim=True) # summing up the rows, keepdim=True to keep the dimensions same
P.shape


g = torch.Generator().manual_seed(2147483647)

for i in range(20):
    out = []
    ix = 0
    while True:
        p = P[ix]
        #p = N[ix].float()
        #p = p /p.sum()
        ix = torch.multinomial(p,num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix==0:
            break
    print(''.join(out))



log_likelihood = 0.0
n=0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P [ix1,ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        #print(f'{ch1}{ch2}:{prob:.4f} {logprob:.4f}')
print(f'{log_likelihood:}')
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}') #So the objective of our training is to find the parameters that minimize the log likehlihood loss

#IMP - Goal is to maximize likelihood of the data w.r.t. model parameters (statistical modeling)
# Equivalent to maximizing log likelihood (because log is monotonically increasing function)
# equivalent to minimizing negative log likelihood
# equivalent to minimizing the average negative log likelihood

# log(a*b*c) = log(a) + log(b) + log(c)

# model smoothing add an integer such as P = (N+1).float() or P = (N+1000).float() the more you add the more smoother uniform model you have it also ensures there are no zeros in the probability matrix avoiding infinity log likelihoods


# create the training set of all bigrams (x,y)



for w in words[:1]:
    chs = ['.'] + list(w) + ['.']
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        print(ch1,ch2)
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)


xenc = F.one_hot(xs, num_classes=27).float() #dtype should be float32 instead of int64 for neural networks to take a variety of values
xenc.shape
plt.imshow(xenc)

xenc.dtype

W = torch.randn((27,27))
logits= xenc @ W #consider them as log counts
counts = logits.exp() # equivalent to N
probs = counts / counts.sum(dim=1,keepdim=True) # equivalent to P
probs
probs[0].sum()


# Forward pass


g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), generator=g)


xenc = F.one_hot(xs, num_classes=27).float()
logits= xenc @ W #consider them as log counts
counts = logits.exp() # equivalent to N
probs = counts / counts.sum(dim=1,keepdim=True) # equivalent to P
# the above two lines are softmax 

nlls = torch.zeros(5)
for i in range(5):
  # i-th bigram:
  x = xs[i].item() # input character index
  y = ys[i].item() # label character index
  print('--------')
  print(f'bigram example {i+1}: {itos[x]}{itos[y]} (indexes {x},{y})')
  print('input to the neural net:', x)
  print('output probabilities from the neural net:', probs[i])
  print('label (actual next character):', y)
  p = probs[i, y]
  print('probability assigned by the net to the the correct character:', p.item())
  logp = torch.log(p)
  print('log likelihood:', logp.item())
  nll = -logp
  print('negative log likelihood:', nll.item())
  nlls[i] = nll

print('=========')
print('average negative log likelihood, i.e. loss =', nlls.mean().item())


xs = []
ys = []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples:', num)



g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), generator=g, requires_grad=True)

for k in range(100):

    #forward pass
    xenc = F.one_hot(xs, num_classes=27).float()
    logits= xenc @ W #consider them as log counts
    counts = logits.exp() # equivalent to N
    probs = counts / counts.sum(dim=1,keepdim=True) # equivalent to P
    # the above two lines are softmax 

    loss=-probs[torch.arange(num),ys].log().mean() + 0.01*(W**2).mean()# vectorized version
    loss
    print(loss.item())

    #backward pass
    W.grad = None # set to zero the gradient
    loss.backward()
    W.grad

    # update the tensor
    W.data += -50*W.grad

# incentivizing W to be close to zero makes it smooth/ uniform

# finally, sample from the 'neural net' model
g = torch.Generator().manual_seed(2147483647)

for i in range(5):
  
  out = []
  ix = 0
  while True:
    
    # ----------
    # BEFORE:
    #p = P[ix]
    # ----------
    # NOW:
    xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
    logits = xenc @ W # predict log-counts
    counts = logits.exp() # counts, equivalent to N
    p = counts / counts.sum(1, keepdims=True) # probabilities for next character
    # ----------
    
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))