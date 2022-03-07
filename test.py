import numpy as np
A = np.random.random((3, 4))
B = np.random.random((4, 3))

# 수동계산
C = np.array([[sum(r*c for r, c in zip(row,col)) for col in zip(*B)] for row in A]).reshape(3, 3)
print('manual C is')
print(C)

# numpy 사용
np_C = np.dot(A,B)
print('np_C is')
print(np_C)

# 비교
print('is it same?')
print(C == np_C)