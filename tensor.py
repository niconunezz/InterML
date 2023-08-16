from InterML import inter
from typing import Tuple
from value import Value


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

