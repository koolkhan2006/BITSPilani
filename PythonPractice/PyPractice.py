#maps, hashmaps, lookup tables, or associative arrays
phonebook = {
     "bob": 7387,
     "alice": 3719,
     "jack": 7052
 }
print(phonebook["alice"])
phonebook["alice"] = "1234"
print(phonebook)
print(phonebook.keys())
squares = {x: x * x for x in range(6)}

squares[6] = 16
print(squares)

#Ordered maps, hashmaps, lookup tables, or associative arrays
import collections
d = collections.OrderedDict(one=1, two=2, three=3)
d["four"] = 4
print(d)
print(d.keys())


#Arrays
# Initilaizing 2D array to store the vertices
arr_len = 8
rows, cols = (arr_len, arr_len)
arr = [[0 for i in range(cols)] for j in range(rows)]

print("\nThe 2D-Array is:")
for i in arr:
    for j in i:
        print(j, end="\t\t")
    print()

#Lists
arr = ["one", "two", "three"]
print(arr[0])
# Lists have a nice repr:
print(arr)

# Lists are mutable:
arr[1] = "hello"
print(arr)

del arr[1]
print(arr)

# Lists can hold arbitrary data types:
arr.append(23)
print(arr)

arr = ("one", "two", "three")
arr[0]

# Tuples have a nice repr:
print(arr)

# Tuples are immutable:
#arr[1] = "hello"
#del arr[1]

# Tuples can hold arbitrary data types:
# (Adding elements creates a copy of the tuple)
print(arr.__hash__())
arr = arr + (23,)
print(arr.__hash__())

#Stack
s = []
s.append("eat")
s.append("sleep")
s.append("code")

print(s)
print(s.pop())
print(s.pop())
print(s.pop())
#s.pop()

from queue import Queue
q = Queue()
q.put("eat")
q.put("sleep")
q.put("code")
print(q)
print(q.get())
print(q.get())
print(q.get())


from multiprocessing import Queue
q = Queue()
q.put("eat")
q.put("sleep")
q.put("code")
print(type(q))
q.get()
q.get()
q.get()

from queue import PriorityQueue
q = PriorityQueue()
q.put((2, "code"))
q.put((1, "eat"))
q.put((4, "sleep"))
q.put((3, "play Cricket"))

while not q.empty():
     next_item = q.get()
     print(next_item)
