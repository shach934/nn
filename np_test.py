import numpy as np

# np random need number input 
# np zeros need tuple input

a = np.random.randn(10,1)

print(a.shape)
print(sum(a))

a = np.array([[1,2,3],[4,5,6]])
print(a.sum())
print(np.sum(a, axis= 0))
print(a / np.sum(a, axis = 0))

b = np.zeros((10,1))

c = np.array([[1,2,3]])
print(c)

print(c.shape)

print(c[0, 0])

d = [4,5,6]

d = np.array( d + d)

np.reshape(d, (2, 3))

print(d)
print(d.shape)

index = np.array([1,5])
d[index] = 0
print(d)