from InterML import inter
from typing import Tuple
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
      self.grad = 1.0 * out.grad
      other.grad = 1.0 * out.grad
    out._backward = _backward

    return out

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.val * other.val, (self, other), '*')

    def _backward():
      self.grad += (other.val) * out.grad
      other.grad += (self.val) * out.grad
    out._backward = _backward

    return out

  def __sub__(self,other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.val - other.val, (self, other), '-')

    def _backward():
      self.grad = -1.0 * out.grad
      other.grad = -1.0 * out.grad
    out._backward = _backward

    return out

  def reLU(self):
    out = Value(max(0,self.val), (self,), 'ReLU')
    def _backward():
      self.grad = float(out.val > 0) * out.grad
    out._backward = _backward
    return out

  def tanh(self):
    e = 2.718281828459045
    numerator = (e ** self.val) - (e ** (-self.val))
    denominator = (e ** self.val) + (e ** (-self.val))
    out = Value(numerator/denominator, (self,), 'tanh')

    def _backward():
      self.grad = 1.0 - out.val ** 2
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


# a tensor implementation, used for matrix multiplication and that type of stuff

class Tensor():
  def __init__(self,val):
    assert isinstance(val,list),'val must be a list'
    self.val = val
  
  def __getitem__(self,index):
    return self.val[index]
  
  def __add__(self,other):
    assert isinstance(other,Tensor), 'other must be a tensor'
    assert self.ndim() == other.ndim(), 'tensor must have the same number of dimensions'

    def add_matrices(matrix1, matrix2):
      if isinstance(matrix1[0], list):
          result = [add_matrices(row1, row2) for row1, row2 in zip(matrix1, matrix2)]
      else:
          result = [x + y for x, y in zip(matrix1, matrix2)]
      return result
    return Tensor(add_matrices(self.val,other.val))

  def __mul__(self,other):
    
    assert isinstance(other,Tensor), 'other must be a tensor'
    assert self.ndim() == other.ndim(), 'tensor must have the same number of dimensions'
    def mul_matrices(matrix1, matrix2):
      if isinstance(matrix1[0], list):
          result = [mul_matrices(row1, row2) for row1, row2 in zip(matrix1, matrix2)]
      else:
          result = [x*y for x, y in zip(matrix1, matrix2)]
      return result
    return Tensor(mul_matrices(self.val,other.val))

  def backward(self):
    
    def activate_all(tensor):
      if isinstance(tensor[0], list):
          return [activate_all(row) for row in tensor]
      else:
          for i in tensor:
              i.backward()
    activate_all(self)

  def dim(self):
    def get_dimensions_lengths(self):
      dimensions_lengths = []

      if isinstance(self, list):
    
          dimensions_lengths.append(len(self))
          if len(self) > 0:
              inner_lengths = get_dimensions_lengths(self[0])
              dimensions_lengths.extend(inner_lengths)

      return dimensions_lengths
    return get_dimensions_lengths(self.val)
  
  def ndim(self):
    return len(self.dim())

  def __repr__(self):
    return 'Tensor(' + str(self.val) + ')'

