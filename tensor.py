from InterML import inter
from typing import Tuple
from value import Value
from typing import List
from tools import all_together

# a tensor implementation, used for matrix multiplication and that type of stuff

class Tensor():
  def __init__(self,val):
    assert isinstance(val,list),'val must be a list'
    self.flow(val)
    self.val = val
  
  # converts everithing in the tensor in values if they are not already
  def flow(self,tens: List):
    if isinstance(tens[0], list):
        for i in range(len(tens)):
            self.flow(tens[i])
    else:
        for i in range(len(tens)):
            tens[i] = tens[i] if isinstance(tens[i], Value) else Value(tens[i])

  
  def __getitem__(self,index):
    return self.val[index]
  
  def sum(self):
    def funct(tens):
      all = 0
      for row in tens:
        if isinstance(row, list):
            all += funct(row)
        else:
            all += row
      return all
    return funct(self.val)
  
  def mean(self):
    total_num= 1
    for i in range(self.ndim()):
      total_num *= self.size()[i]

    return self.sum() *  (total_num**-1)
  
  
  def __add__(self,other):
    assert isinstance(other,Tensor) or isinstance(other,list)
    other = other if isinstance(other, Tensor) else Tensor(other)
    assert self.ndim() == other.ndim(), 'tensor must have the same number of dimensions'

    def add_matrices(matrix1, matrix2):
      if isinstance(matrix1[0], list):
          result = [add_matrices(row1, row2) for row1, row2 in zip(matrix1, matrix2)]
      else:
          result = [x + y for x, y in zip(matrix1, matrix2)]
      return result
    return Tensor(add_matrices(self.val,other.val))
  
  def __radd__(self,other):
    return self + other

  def __mul__(self,other):
    assert isinstance(other,Tensor) or isinstance(other,list)
    other = other if isinstance(other, Tensor) else Tensor(other)
    assert self.ndim() == other.ndim(), 'tensor must have the same number of dimensions'

    def mul_matrices(matrix1, matrix2):
      if isinstance(matrix1[0], list):
          result = [mul_matrices(row1, row2) for row1, row2 in zip(matrix1, matrix2)]
      else:
          result = [x*y for x, y in zip(matrix1, matrix2)]
      return result
    return Tensor(mul_matrices(self.val,other.val))
  
  def __rmul__(self,other):
     return self * other

  def __matmul__(self,other):
     assert isinstance(other,Tensor), 'other must be a tensor'
     assert self.size()[-1] == other.size()[-2], 'dimensions must match'
     global matrices
     matrices = []
     out = all_together(self,other)
     return Tensor(out)
  
  def __pow__(self,other):
    assert isinstance(other, int), "only supporting int powers for now"
    incremental = self
    for _ in range(other-1):
      incremental =  incremental * self
    return incremental
  
  def reLU(self):
    def funct(tens):
      if isinstance(tens[0], list):
          for i in range(len(tens)):
              funct(tens[i])
      if isinstance(tens[0], Value):
          for i in range(len(tens)):
              tens[i] = tens[i].reLU()
    return funct(self.val)
  
  def tanh(self):
    def funct(tens):
      if isinstance(tens[0], list):
          for i in range(len(tens)):
              funct(tens[i])
      if isinstance(tens[0], Value):
          for i in range(len(tens)):
              tens[i] = tens[i].tanh()
    return funct(self.val)
  
  def backward(self):
    
    def activate_all(tensor):
      if isinstance(tensor[0], list):
          return [activate_all(row) for row in tensor]
      else:
          for i in tensor:
              i.backward()
    activate_all(self)
  

  def size(self):
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
    return len(self.size())

  def __repr__(self):
    return 'Tensor(' + str(self.val) + ')'

