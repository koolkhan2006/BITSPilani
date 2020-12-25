import itertools

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
f.write("Total no. of freight trains:" + len(uniq_trains).__str__())
f.write("\n")
f.write("Total no. of cities:" + len(uniq_cities).__str__())
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


def populate_matrix(city_1,city_2):
    if arr[uniq_cities.index(city_1)][uniq_cities.index(city_2)] == 0:
        arr[uniq_cities.index(city_1)][uniq_cities.index(city_2)] = train
        arr[uniq_cities.index(city_2)][uniq_cities.index(city_1)] = train
    else:
        arr[uniq_cities.index(city_1)][uniq_cities.index(city_2)] = arr[uniq_cities.index(city_1)][
                                                                        uniq_cities.index(city_2)] + "," + train
        arr[uniq_cities.index(city_2)][uniq_cities.index(city_1)] = arr[uniq_cities.index(city_2)][
                                                                        uniq_cities.index(city_1)] + "," + train


for x in f:
  i =0
  input = x.split("/")
  train = input[i].strip()
  input.remove(input[i])
  if len(input) == 2:
    city_1 = input[i].strip()
    city_2 = input[i+1].strip()
    populate_matrix(city_1,city_2)
  elif len(input) > 2:
      for x in itertools.combinations(input, 2):
          city_1 = x[i].strip()
          city_2 = x[i + 1].strip()
          populate_matrix(city_1, city_2)


print("\nThe 2D-Array is:")
for i in arr:
    for j in i:
        print(j, end="\t\t")
    print()

for i in range(len(arr)):
    for j in range(i):
        print(arr[i][j], end="\t\t")
    print()

max_count = 0
for i in range(len(arr)):
    tmp_list_trains = list()
    for j in range(len(arr)):
        if(arr[i][j] != 0):
            tmp_list_trains.extend(str(arr[i][j]).split(','))
    curr_count = len(unique(tmp_list_trains))
    if(curr_count > max_count):
        max_index = i
        list_trains = unique(tmp_list_trains)
        max_count = curr_count

print(uniq_cities[max_index])
print(list_trains)

f = open("outputPS22.txt", "a")
f.write("--------Function displayTransportHub --------")
f.write("\n")
f.write("Main transport hub:" + uniq_cities[max_index].__str__())
f.write("\n")
f.write("Number of trains visited:" + len(list_trains).__str__())
f.write("\n")
f.write("List of Freight trains:")
f.write("\n")
for x in list_trains:
    f.write(x)
    f.write("\n")
f.close()

# function to check if small string is
# there in big string
def check(string, sub_str):
    if (string.find(sub_str) == -1):
        return False
    else:
        return True

list_of_connected_cities = list()
for i in range(len(arr)):
    for j in range(i):
        if(check(str(arr[i][j]),'T1122')):
            list_of_connected_cities.append(uniq_cities[i])
            list_of_connected_cities.append(uniq_cities[j])

print(unique(list_of_connected_cities))
f = open("outputPS22.txt", "a")
f.write("--------Function displayConnectedCities --------")
f.write("\n")
f.write("Freight train number:" + 'T1122')
f.write("\n")
f.write("Number of cities connected:" + len(unique(list_of_connected_cities)).__str__())
f.write("\n")
f.write("List of cities connected directly by T1122:")
f.write("\n")
for x in unique(list_of_connected_cities):
    f.write(x)
    f.write("\n")
f.close()


city_1 = 'Vishakhapatnam'
city_2 = 'Hyderabad'
package_can_be_sent = 'No, Apology no direct trains available on this route'
if(arr[uniq_cities.index(city_1)][uniq_cities.index(city_2)] != 0):
    print(arr[uniq_cities.index(city_1)][uniq_cities.index(city_2)])
    package_can_be_sent = 'Yes, Package can be sent through '+arr[uniq_cities.index(city_1)][uniq_cities.index(city_2)]

print(package_can_be_sent)

f = open("outputPS22.txt", "a")
f.write("--------Function displayDirectTrain --------")
f.write("\n")
f.write("City A:" + city_1.__str__())
f.write("\n")
f.write("City B:" + city_2.__str__())
f.write("\n")
f.write("Package can be sent directly:" + package_can_be_sent)
f.close()