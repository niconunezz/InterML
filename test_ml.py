import pytest
from tensor import Tensor



class TestClass:
    m1 = Tensor([[1, 2, 3], [4, 5, 6]])
    m2 = Tensor([[1, 5, 6], [4, 3, 6]])
    value = 0
    def test_addition(self):
        result = self.m1 + self.m2
        expected = Tensor([[2, 7, 9], [8, 8, 12]])
        assert result.val == expected.val
    
    def test_mul(self):
        result = self.m1 * self.m2
        expected = Tensor([[1, 10, 18], [16, 15, 36]])
        assert result.val == expected.val
    
