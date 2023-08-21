import pytest
from tensor import Tensor

class TestClass:
    m1 = Tensor([[1, 2, 3], [4, 5, 6]])
    m2 = Tensor([[1, 5, 6], [4, 3, 6]])
    matrix_a = Tensor([
    [[[1, 2, 4],
    [3, 4, 6],
    [5, 6, 8]],
    [[1, 2, 4],
    [3, 4, 6],
    [5, 6, 8]]],
    [[[1, 2, 4],
    [3, 4, 6],
    [5, 6, 8]],
    [[1, 2, 4],
    [3, 4, 6],
    [5, 6, 8]]]
])
    
    matrix_b =Tensor( [
   [[[1, 2],
    [3, 4],
    [5, 6]],
   [[1, 2],
    [3, 4],
    [5, 6]]],
    [[[1, 2],
    [3, 4],
    [5, 6]],
   [[1, 2],
    [3, 4],
    [5, 6]]]

])

    def test_flow(self):
        m1 = Tensor([[1, 2, 3], [4, 5, 6]])
        assert m1.val[0][0].type() == 'Value'

    def test_addition(self):
        result = self.m1 + self.m2
        expected = Tensor([[2, 7, 9], [8, 8, 12]])
        expected=[[v.val for v in row] for row in expected.val]
        result = [[v.val for v in row] for row in result.val]
        assert result == expected
    
    def test_radd(self):
        result = [[1, 2, 3], [4, 5, 6]] + self.m2
        expected = Tensor([[2, 7, 9], [8, 8, 12]])
        expected=[[v.val for v in row] for row in expected.val]
        result = [[v.val for v in row] for row in result.val]
        assert result == expected

    
    def test_mul(self):
        result = self.m1 * self.m2
        expected = Tensor([[1, 10, 18], [16, 15, 36]])
        expected=[[v.val for v in row] for row in expected.val]
        result = [[v.val for v in row] for row in result.val]
        assert result == expected
    
    def test_rmul(self):
        result = [[1, 2, 3], [4, 5, 6]] * self.m2
        expected = Tensor([[1, 10, 18], [16, 15, 36]])
        expected=[[v.val for v in row] for row in expected.val]
        result = [[v.val for v in row] for row in result.val]
        assert result == expected
    
    
    def test_matmul4dim(self):

        print('matrix_a',self.matrix_a.size())
        print('matrix_b',self.matrix_b.size())
        result = self.matrix_a @ self.matrix_b
        print(result)
        expected = Tensor([[[[27, 34], [45, 58], [63, 82]], [[27, 34], [45, 58], [63, 82]]], [[[27, 34], [45, 58], [63, 82]], [[27, 34], [45, 58], [63, 82]]]])
        print(expected.size())
        expected_values = [[[[v.val for v in inner_row] for inner_row in row] for row in matrix] for matrix in expected.val]
        result_values = [[[[v.val for v in inner_row] for inner_row in row] for row in matrix] for matrix in result.val]
        assert result_values == expected_values




    




    
    
    
