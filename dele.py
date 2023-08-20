# just a file where i try some code :)
from tensor import Tensor
matrix_a = [[1, 2, 3], [4, 5, 6]]

matrix_b = [[1, 2],[3, 4],[5, 6]]

a = Tensor(matrix_a)
b = Tensor(matrix_b)

print(a @ b)
