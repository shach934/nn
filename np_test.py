import numpy as np

# np random need number input 
# np zeros need tuple input

a = np.random.randn(10,1)
print(a)
print(sum(a))

b = np.zeros((10,1))
print(b)

c = np.array([[1,2,3]])
print(c)

print(c.shape)

print(c[0, 0])

d = [4,5,6]

d = np.array([d,  d])

print(sum(sum(d)))

print(d)
print(d.shape)


e = d > 2
print(d + e)

