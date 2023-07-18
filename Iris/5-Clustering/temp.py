import numpy as np

A = np.array([[1,0.191,0.995],[0.191,1,0.295],[0.995,0.295,1]])
# print(A)
# u,s,v = np.linalg.svd(A)
# print(u,s,v)
e, v = np.linalg.eig(A)
print(e,v)