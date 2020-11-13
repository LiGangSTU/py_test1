import  numpy as np


a = np.array([1,2,3])
print(a)

b = np.array([1,2,3],dtype=complex)
print(b)

c = np.dtype(np.int32)
print(c)

student  = np.dtype([('name','S20'),('age','i1')])
a = np.array([('abc',21),('xyz',18)],dtype=student)
print(a)

a = np.array([[1,2,3],[4,5,6]])
a.shape = (3,2)
print(a)
a.reshape(1,6)
print(a)