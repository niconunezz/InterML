# My way of keeping track of values gradients
class Value():
  def __init__(self,val:float,_children=(),_op=''):
    self.val = val
    self.grad = 0
    self._prev = set(_children)
    self._backward = lambda: None

  def __repr__(self):
    return 'Value(' + str(self.val) + ')'

  def __add__(self, other):

    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.val + other.val, (self, other), '+')


    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward

    return out
  
  def __radd__(self, other):
    return self + other

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.val * other.val, (self, other), '*')

    def _backward():
      self.grad += (other.val) * out.grad
      other.grad += (self.val) * out.grad
    out._backward = _backward

    return out
  
  def __rmul__(self, other):
    return self * other

  def __sub__(self,other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.val - other.val, (self, other), '-')

    def _backward():
      self.grad += -1.0 * out.grad
      other.grad += -1.0 * out.grad
    out._backward = _backward

    return out

  def __rsub__(self,other):
    return self - other

  def reLU(self):
    out = Value(max(0,self.val), (self,), 'ReLU')
    def _backward():
      self.grad += float(out.val > 0) * out.grad
    out._backward = _backward
    return out

  def tanh(self):
    e = 2.718281828459045
    numerator = (e ** self.val) - (e ** (-self.val))
    denominator = (e ** self.val) + (e ** (-self.val))
    out = Value(numerator/denominator, (self,), 'tanh')

    def _backward():
      self.grad += 1.0 - out.val ** 2
    out._backward = _backward
    return out

  def backward(self):
    topo = []
    visited = set()
    def build_top(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_top(child)
        topo.append(v)
    build_top(self)
    self.grad = 1.0
    for node in reversed(topo):
      node._backward()