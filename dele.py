# just a file where i try some code :)
from tensor import Tensor
matrix_a = [
[    [[1, 2, 4],
    [3, 4, 6],
    [5, 6, 8]],

    [[1, 2, 6],
    [3, 4, 5],
    [5, 6,8]]]
]

matrix_b = [
[    [[1, 2],
    [3, 4], 
    [5, 6]],

    [[1, 2],
    [3, 4],
    [5, 6]]]
]

a = Tensor(matrix_a)
b = Tensor(matrix_b)

print(a.size()[-1])
print(b.size()[-2])

