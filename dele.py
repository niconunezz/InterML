# just a file where i try some code :)
from tensor import Tensor
from typing import List
from value import Value

m1 = Tensor([[1, 2, 3], [4, 5, 6]])
m2 = Tensor([[1, 5, 6], [4, 3, 6]])

print(m1.val[0][0].type())

