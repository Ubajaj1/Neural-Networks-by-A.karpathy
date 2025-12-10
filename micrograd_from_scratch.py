import math
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import random

def f(x):
    return 3*x**2 - 4*x + 5

#f(3.0)

xs = np.arange(-5,5,0.25)
ys=f(xs)
plt.plot(xs,ys)

h=0.001
x=3.0
(f(x+h)-f(x))/h

### more complex
a=2.0
b=-3.0
c=10.0
d= a*b + c
print(d)

h=0.0001

# inputs
a=2.0
b=-3.0
c=10.0

d1 = a*b + c
b+=h
d2 = a*b + c

print('d1',d1)
print('d2',d2)
print('slope', (d2-d1)/h)

### creation of value object

class Value:

    def __init__(self,data,_children=(), _op='', label=''):
        self.data=data
        self.grad=0.0
        self._backward=lambda: None
        self._prev=set(_children)
        self._op=_op
        self.label=label
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self,other):
        other= other if isinstance(other,Value) else Value(other)
        out=Value(self.data+other.data,(self,other),'+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __mul__(self,other):
        other= other if isinstance(other,Value) else Value(other)
        out=Value(self.data*other.data,(self,other),'*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self,other):
      assert isinstance(other,(int,float)),"only supporting int/float powers for now"
      out=Value(self.data**other,(self,),f'**{other}')

      def _backward():
        self.grad += other * (self.data ** (other - 1)) * out.grad
      out._backward = _backward
      return out

    def __rmul__(self,other):
        return self*other
    
    def __truediv__(self,other):
        return self*other**-1

    def __neg__(self):
        return self*-1
    def __sub__(self,other):
        return self+(-other)


    def tanh(self):
        n = self.data
        t = (math.exp(2*n) - 1) / (math.exp(2*n) + 1)
        out = Value(t,(self,),'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    def exp(self):
        x=self.data
        out=Value(math.exp(x),(self,),'exp')

        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward
        return out
    
    def backward(self):
        
        topo = []
        visited = set()
        def build_topo(v):
          if v not in visited:
            visited.add(v)
            for child in v._prev:
              build_topo(child)
            topo.append(v)
        build_topo(self)
        topo
        self.grad=1.0
        for node in reversed(topo):
          node._backward()

a=Value(2.0,label='a')
b=Value(-3.0,label='b')
c=Value(10.0,label='c')
e= a*b; e.label='e'
d= e+c; d.label='d'
f= Value(-2.0,label='f')
L = d*f; L.label='L'
L


from graphviz import Digraph

def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot

draw_dot(L)

L.grad=1.0
f.grad=4.0
d.grad=-2.0
c.grad=-2.0
e.grad=-2.0
a.grad = 6.0
b.grad = -4.0

# need to calculate dL/dc
#d= c + e
#dd/dc = 1
#dd/de = 1

##chain rule

#dL/dc = (dl/dd) * (dd/dc)


#dL/da = (dL/de)*(de/da)



def lol():
    h = 0.0001
    a=Value(2.0,label='a')
    b=Value(-3.0,label='b')
    c=Value(10.0,label='c')
    e= a*b; e.label='e'
    d= e+c; d.label='d'
    f= Value(-2.0,label='f')
    L = d*f; L.label='L'
    L1=L.data

    a=Value(2.0,label='a')
    b=Value(-3.0,label='b')
    c=Value(10.0,label='c')
    e= a*b; e.label='e'
    d= e+c; d.label='d'
    f= Value(-2.0,label='f')
    L = d*f; L.label='L'
    L2=L.data+h

    print((L2-L1)/h)

lol()

# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'
o.backward()
draw_dot(o)

# Calculate gradients
# o = tanh(n)
# do/dn  = 1 - tanh(n)**2 = 1 - o**2
#1 - o.data**2

o.grad=1.0
o._backward()
n._backward()
x1w1x2w2._backward()
b._backward()
x2w2._backward()
x1w1._backward()

n.grad=0.5
x1w1x2w2.grad=0.5
b.grad=0.5
x2w2.grad=0.5
x1w1.grad=0.5
x2.grad = w2.data * x2w2.grad
w2.grad = x2.data * x2w2.grad
x1.grad = w1.data * x1w1.grad
w1.grad = x1.data * x1w1.grad

# topological sort (edges go from left to right in one direction)
topo = []
visited = set()
def build_topo(v):
  if v not in visited:
    visited.add(v)
    for child in v._prev:
      build_topo(child)
    topo.append(v)
build_topo(o)
topo

for node in reversed(topo):
  node._backward()


# issue with this approach is that it will fail anytime when we have a variable mentioned twice in the expression
# for example, if we have a*b + a*b, it will fail
# to fix this, we need to use multivariate case where we need to accumulate the gradients and not directly set them


# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
e = (2*n).exp(); e.label = 'e'
o = (e-1)/(e+1);o.label = 'o'
o.backward()
draw_dot(o)

import torch

x1 = torch.Tensor([2.0]).double()  ;x1.requires_grad=True
x2 = torch.Tensor([0.0]).double()  ;x2.requires_grad=True
w1 = torch.Tensor([-3.0]).double() ;w1.requires_grad=True
w2 = torch.Tensor([1.0]).double() ;w2.requires_grad=True
b = torch.Tensor([6.8813735870195432]).double() ;b.requires_grad=True
n= x1*w1 + x2*w2 + b
o = torch.tanh(n)

print(o.data.item())
o.backward()

print('---')
print('x2',x2.grad.item())
print('w1',w2.grad.item())
print('x1',x1.grad.item())
print('w1',w1.grad.item())

class Neuron:

  def __init__(self, nin):
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))
  
  def __call__(self, x):
    act =sum((wi*xi for wi,xi in zip(self.w,x)),self.b) #for notations n(x), self.b is starter variable in sum
    out = act.tanh()
    return out
  
  def parameters(self):
    return self.w + [self.b]

class Layer:

  def __init__(self, nin, nout): #nin is number of inputs, nout is number of neurons in a single laye
    self.neurons = [Neuron(nin) for _ in range(nout)]

  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:

  def __init__(self, nin, nouts): #nin is number of inputs, nouts is list of sizes of layer in MLP
    sz = [nin] + nouts
    self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(nouts))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]



x = [2.0,3.0,-1.0]

n = MLP(3,[4,4,1])

n(x)

len(n.parameters())

draw_dot(n(x))

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0,1.0,-1.0]
]
ys = [1.0, -1.0, -1.0, 1.0] #desired targets
ypred = [n(x) for x in xs]
ypred

# loss is a single number that measures the performance the whole neural network

loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
loss

for p in n.parameters():
  p.data+= -0.01*p.grad

for k in range(20):
  #forward pass
  ypred = [n(x) for x in xs]
  loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
  print(loss.data)

  #backward pass
  for p in n.parameters():
    p.grad = 0.0
  loss.backward()

  #update
  for p in n.parameters():
    p.data+= -0.05*p.grad
  
  print(k,loss.data)