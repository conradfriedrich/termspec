# -*- coding: utf-8 -*-


from scipy.spatial.distance import cosine

a = (1,0,0,1.4,1,3,2)
b = (2,15,2,0,0,1,3)


# c = (0,0,1,0,0,1)
# d = (1,1,0,1,1,0)

# print(1 - cosine(c,d)) # should be 0 ?
# print(1 - cosine(c,c)) # should be 1 ?

print(1 - cosine(a,b))


a = (1,0,0,0,0,0,0,0,0,0,0,1.4,1,3,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
b = (2,15,2,0,0,0,0,0,0,0,0,0,0,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)

print(1 - cosine(a,b))


a = np.array([1,5])
b = np.array([1,0])

print(cosine(a,b))
print( a)
print( b)
# print(1 - cosine(X[5],X[2]))
# print( X[5])
# print( X[2])

c = np.array([3,5])
d = np.array([1,0])
print(cosine(c,d))
print( c)
print( d)
# print(1 - cosine(X[6],X[5]))
# print( X[6])
# print( X[5])