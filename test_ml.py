import pytest
from tensor import Tensor

class TestClass:
    m1 = Tensor([[1, 2, 3], [4, 5, 6]])
    m2 = Tensor([[1, 5, 6], [4, 3, 6]])
    value = 0
    def test_addition(self):
        result = self.m1 + self.m2
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
    
    def test_flow(self):
        m1 = Tensor([[1, 2, 3], [4, 5, 6]])
        assert m1.val[0][0].type() == 'Value'
