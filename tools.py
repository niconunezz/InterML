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

matrices = []
def all_together(tensor_a, tensor_b):
    andim = tensor_a.ndim()
    bndim = tensor_a.ndim()
    matrix_a = tensor_a.val
    matrix_b = tensor_b.val

    find2dim(matrix_a)
    find2dim(matrix_b)
    n = len(matrices)//2
    
    a = matrices[:n]
    b = matrices[n:]
    if andim == 4:
      newa = []
      newb = []
      fdim = tensor_a.size()[0]
      n = len(a)//fdim
      j,k=0,n
      for i in range(fdim):
        newa.append(a[j:k])
        newb.append(b[j:k])
        j,k = k,k+n
      first_step = []
      out = []
      for fda,fdb in zip(newa,newb):
        for x,y in zip(fda,fdb):
          first_step.append(matrix_multiplication(x,y))
        out.append(first_step)
        first_step = []
  
    else:
      out= []
      for m1,m2 in zip(a,b):
        out.append(matrix_multiplication(m1,m2))

    new = out

    return new



