f = open("inputPS22.txt", "r")
list_of_trains = list()
list_of_cities = list()

# function to get unique values
def unique(list1):
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    return unique_list

for x in f:
  i =0
  input = x.split("/")
  list_of_trains.append(input[i].strip())
  input.remove(input[i])
  for y in input:
      list_of_cities.append(y.strip())

uniq_trains = unique(list_of_trains)
uniq_cities = unique(list_of_cities)
print(uniq_trains)
print(uniq_cities)

f = open("outputPS22.txt", "w")
f.write("--------Function showAll --------")
f.write("\n")
f.write("Total no. of freight trains:" + len(list_of_trains).__str__())
f.write("\n")
f.write("Total no. of cities:" + len(list_of_cities).__str__())
f.write("\n")
f.write("List of Freight trains:")
f.write("\n")
for x in uniq_trains:
    f.write(x)
    f.write("\n")
f.write("List of cities:")
f.write("\n")
for x in uniq_cities:
    f.write(x)
    f.write("\n")
f.close()

# Initilaizing 2D array to store the vertices
arr_len = len(uniq_cities)
rows, cols = (arr_len, arr_len)
arr = [[0 for i in range(cols)] for j in range(rows)]

f = open("inputPS22.txt", "r")
for x in f:
  i =0
  input = x.split("/")
  train = input[i].strip()
  input.remove(input[i])
  city_1 = input[i].strip()
  city_2 = input[i+1].strip()
  if arr[uniq_cities.index(city_1)][uniq_cities.index(city_2)] == 0:
      arr[uniq_cities.index(city_1)][uniq_cities.index(city_2)] = train
  else:
      arr[uniq_cities.index(city_1)][uniq_cities.index(city_2)] = arr[uniq_cities.index(city_1)][uniq_cities.index(city_2)] + "," + train

print(arr)
