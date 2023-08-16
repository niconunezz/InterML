from tensor import Tensor

def matrix_multiplication(matrix_a, matrix_b):
    def multiply_recursive(matrix1, matrix2):
        if isinstance(matrix1[0], list):
            result = [
                [0 for _ in range(len(matrix2[0]))]
                for _ in range(len(matrix1))
            ]
            for i in range(len(matrix1)):
                for j in range(len(matrix2[0])):
                    for k in range(len(matrix1[0])):
                        result[i][j] += matrix1[i][k] * matrix2[k][j]
            return result
        else:
            return matrix1 * matrix2

    result = multiply_recursive(matrix_a, matrix_b)
    
    return result

def find2dim(matrix):
  if isinstance(matrix, list):
    for m in range(len(matrix)):
      if isinstance(matrix[m], list):
        if isinstance(matrix[m][0],list):
          find2dim(matrix[m])
        else:
          matrices.append(matrix)
          
          return False

matrix_a=[[[ 29,  87],
         [109, 106]],

        [[ 40,  98],
         [107,  27]],
          
        [[ 40,  98],
         [107,  27]]]
matrix_b = [[[37, 28],
         [31, 13]],

        [[30,  8],
         [ 8, 37]],
            
        [[30,  8],
         [ 8, 37]]]   

matrices = []
def all_together(matrix_a, matrix_b):
  find2dim(matrix_a)
  find2dim(matrix_b)
  n = len(matrices)//2

  a = matrices[:n]
  b = matrices[n:]
  
  
  out= []
  for m1,m2 in zip(a,b):
    out.append(matrix_multiplication(m1,m2))

  new = Tensor(out)

  return new

print(all_together(matrix_a, matrix_b).size())