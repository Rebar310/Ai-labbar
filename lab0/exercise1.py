import numpy as np

#x = np.arange(0, 1, 0.01)
#print(x)

# Exercise 1 ------------------------------------
print((1, 2)) # wrong this is not a list with two elements
# a list with two elements looks like [1, 2]
print([1,2])

print((1, 2)) # yes, it is a tuple with two elements

print({}) # wrong, this is an empty dictionary not a set
# a set looks like set()
print(set())

print([1, 2] + [3, 4]) # wrong this will not become [4,6]
# if we write [1+3,2+4] we will get the right answer
print([1+3,2+4])

#txt = "a" + 1000 # wrong becuse of addition of string and int does not work.
# we need to write
txt_1 = "a" * 1000
#print(txt)
print(txt_1)

lst = [1, 2]
lst.append([3, 4]) #This will not become [1, 2, 3 4],
# we need to write extend instead of append
print(lst)

print([[]] * 3) #This is true

X = 10
print(r"x is equal to {x}") # This does not work, instead we write:
print(f"x is equal to {X}") # r is for raw string but we need f for formated string and also a big X

print("ABCD"[::-1]) # This is correct
# the kolon in the brackets stand for [start:stop:step], -1 means go backwards

print("ABCDEF"[1:-1]) #This will not become BCDEF , we have [start,stop]
print("ABCDEF"[1:]) # before we had -1 at the stop witch made in not go to the end of the string

print("12345678"[:2:] == "2468")  # This becomes false not true
#[:2:] says that we need to stop before index 2 so 12 is not 2468
print("12345678"[1::2] == "2468") # start at index 1 go to the end with step 2

print(len({1, 1, 2}) == 3) #This does not become true, a set does not have double of the same value,therefore length 2 is not equal to 3
print(len([1, 1, 2]) == 3) #lists can take two of the same numbers

print(dict()) # Yes this is true, it creates an empty dict

print(r"a\nb") # This will not set b to a new row becuse of r", raw string will think of the n as a part of the string
print("a\nb") # This will give the right result

print("a""b") #This will not print a"b , how we can change this is:
print("a\"b") # We need a backslash before so it knows that it should also show

print("\\\\") #This does not become \\\\, it becomes \\ becuse a backslash menas that the backslash after that will show
print("\\\\\\\\") # Now we will get four backslashes, becuse we have written the double amount

print("a\bc") # True. this becomes c, \b menas erase one backwards

for i in range(2, 5): # this will only write 2 3 4
 print(i) # to get one more number in, the 5 we need to:
for i in range(2, 6): # range stops before the last value wich was 5 before.
    print(i)

#import scipy # this will not import NumPy this imorts SciPy
#import numpy #will import numpy

import numpy as np # "The name numpy is available in the namespace"
# when we right this we use np in the namespace not numpy
# if we would like to use numpy instead we need to just have written import numpy only

from math import sin
from numpy import sin  # "This does not cause any namespace conflicts."
# This is wrong, they will write over each other
# but if we change the namespace so that not both will use sin then the conflict would be resolved
# like this:
# from math import sin
# from numpy import sin as np_sin

#from numpy import * # good practice! (wrong, this is not good practice, very unclear code)

#import numpy as np # good practice! (true)

#from numpy import pi as PI # wrong, when we write like this only PI exists in the namespace
# if we take away "as PI" then pi will be in the namespace

print(np.nan == np.nan) # This is wrong, becuase usually nan != nan and also its bad to use ==
print(np.isnan(np.nan)) # This will become true, because this is how you write equals med numpy

print(1 / np.inf) # Does not become np.nan, instead it becomes 0.0 because 1/infinity becomes zero
# If we want the answer to be np.nan then we need to write
print(np.inf/np.inf) #which becomes nan

print(-1 * np.inf) # this is true it becomes -inf

print(2//3) # Does not become 0.66... becuse of floor divison
print(2/3) # This will get the right result for normal division

#arr = np.zeros((3, 2))
#print(arr.size == 2 ) #Wrong, this is not true
#print(arr.size) # we get a 6 instead of a 2 because that arr.size will count elements wich is 3x2=6
arr_2=np.zeros((1,2))
print(arr_2.size == 2) #This code will therefore become true

#print(0/0) divison with zero does not work directly in python but with Numpy it works
#print(np.float64(0) / np.float64(0)) #Now we get nan

#plt.show() # it will not save the plot to a file instead, show t
#plt.savefig() this will save the file.

print(type(2**3)) # Yes this is true, 2**3= 2^3=8 type(8) = int, becuase no floating numbers

print(np.zeros((3, 2))) # true


#arr == np.zeros((3, 2, 5))) #arr need to be the same size to even be able to make this comparsion ==
#arr.size # arr.size will be 3*2*5 = 30 and not 3,2,5
#if we write arr.shape we will get (3,2,5)

arr = np.zeros((3, 2))
arr_reshaped = arr.reshape((6,))
#arr_reshaped.dim == 1 # The only thing wrong with this code is that there is no attribut that has the name dim
print(arr_reshaped.ndim == 1 ) # This will work now with ndim instead of dim, this will return true

class Book:     #Book is just a class, this is true
    pass        #To create an object we need to write b = Book() and then b will be the object










