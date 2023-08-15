# just like a fast tensor creator

from typing import Tuple
class InterML():

  def ones(self,shape):
    if len(shape) == 0:
        return 1  
    inner_list = [self.ones(shape[1:]) for _ in range(shape[0])]
    return inner_list
  
  def zeros(self,shape):
    if len(shape) == 0:
        return 0
    inner_list = [self.zeros(shape[1:]) for _ in range(shape[0])]
    return inner_list
 
  def fill_tensor(self,value,shape:Tuple):
    if len(shape) == 0:
        return value
    inner_list = [self.fill_tensor(value,shape[1:]) for _ in range(shape[0])]
    return inner_list


inter = InterML()

