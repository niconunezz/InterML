# just a file where i try some code :)
from tensor import Tensor

m1 = Tensor([[1, 2, 3], [4, 5, 6]])
m2 = Tensor([[1, 5, 6], [4, 3, 6]])

result = m1 + m2
expected = Tensor([[2, 7, 9], [8, 8, 12]])

print(result)

assert result.val == expected.val
