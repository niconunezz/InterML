# just like a fast tensor creator

class InterML():

  def __init__(self):
    pass

  def build_ones_tensor(self,shape):
    if len(shape) == 0:
        return 1  
    inner_list = [self.ones(shape[1:]) for _ in range(shape[0])]
    return inner_list
  
  def build_zeros_tensor(self,shape):
    if len(shape) == 0:
        return 0
    inner_list = [self.zeros(shape[1:]) for _ in range(shape[0])]
    return inner_list
  
  def ones(self,shape):
    tensor = self.build_ones_tensor(shape)
    return tensor
  def zeros(self,shape):
    return self.build_zeros_tensor(shape)