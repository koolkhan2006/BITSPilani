#############################################################################
# The Graph class named OnlineRailwayFreightBooking prepares the adjacency matrix
# for the different unique cities and trains running between them.Once the matrix is
# ready all the search operation is done on the matrix.
#############################################################################        
import itertools


# Start of Class OnlineRailwayFreightBooking


class OnlineRailwayFreightBooking:

    def __init__(self):
        self.list_of_trains = []
        self.list_of_cities = []
        self.uniq_trains = []
        self.uniq_cities = []

    # function to get unique values
    @staticmethod
    def unique(list1):
        # insert the list to the set
        list_set = set(list1)
        # convert the set to the list
        unique_list = (list(list_set))
        return unique_list

    # function to initialize 2 dimensional array as per the number of unique trains
    def initializeArray(self):
        # Initializing 2D array to store the vertices
        arr_len = len(self.uniq_cities)
        rows, cols = (arr_len, arr_len)
        self.arr = [[0 for i in range(cols)] for j in range(rows)]
        return self.arr

    def readApplications(self, inputfile):
        """
        Description: Takes the information from "inputfile" and prepare the list of unique trains and cities.
        """
        file = open(inputfile, 'r')
        lines = file.readlines()
        file.close()
        for x in lines:
            i = 0
            input = x.split("/")
            self.list_of_trains.append(input[i].strip())
            input.remove(input[i])
            for y in input:
                self.list_of_cities.append(y.strip())

        self.uniq_trains = self.unique(self.list_of_trains)
        self.uniq_cities = self.unique(self.list_of_cities)

    def showAll(self):
        """
        Description: Outputs all the available information to the output file.
        """
        ## OUTPUT TO FILE - START
        f = open("outputPS22.txt", "w")
        f.write("--------Function showAll --------\n\n")
        f.write("Total no. of freight trains:" + len(self.uniq_trains).__str__())
        f.write("\n")
        f.write("Total no. of cities:" + len(self.uniq_cities).__str__())
        f.write("\n")
        f.write("List of Freight trains:")
        f.write("\n")
        for x in self.uniq_trains:
            f.write(x)
            f.write("\n")
        f.write("\n")
        f.write("List of cities:")
        f.write("\n")
        for x in self.uniq_cities:
            f.write(x)
            f.write("\n")

        f.write("-----------------------------------------\n")
        f.write("\n")
        f.close()
        ## OUTPUT TO FILE - STOP

    def populate_matrix(self, city_1, city_2, train):
        """
        Description: Populate the adjacency matrix by taking 2 input at a time.
        If there are multiple trains running between the 2 cities then it will
        be concatenated with comma and then assigned to specific location in
        matrix.

        """
        if self.arr[self.uniq_cities.index(city_1)][self.uniq_cities.index(city_2)] == 0:
            self.arr[self.uniq_cities.index(city_1)][self.uniq_cities.index(city_2)] = train
            self.arr[self.uniq_cities.index(city_2)][self.uniq_cities.index(city_1)] = train
        else:
            self.arr[self.uniq_cities.index(city_1)][self.uniq_cities.index(city_2)] = \
                self.arr[self.uniq_cities.index(city_1)][
                    self.uniq_cities.index(city_2)] + "," + train
            self.arr[self.uniq_cities.index(city_2)][self.uniq_cities.index(city_1)] = \
                self.arr[self.uniq_cities.index(city_2)][
                    self.uniq_cities.index(city_1)] + "," + train

    def populateAdjacencyMatrix(self):
        """
        Description: Loops over the different cities combination and fills in the adjacency matrix.
        "module itertools" imported to get the different combination in case of single train connecting
        multiple cities.
        """
        self.arr = self.initializeArray()
        f = open("inputPS22.txt", "r")
        for x in f:
            i = 0
            input = x.split("/")
            train = input[i].strip()
            input.remove(input[i])
            if len(input) == 2:
                city_1 = input[i].strip()
                city_2 = input[i + 1].strip()
                self.populate_matrix(city_1, city_2, train)
            elif len(input) > 2:
                for x in itertools.combinations(input, 2):
                    city_1 = x[i].strip()
                    city_2 = x[i + 1].strip()
                    self.populate_matrix(city_1, city_2, train)

    def printAdjacencyMatrix(self):

        for i in self.arr:
            for j in i:
                print(j, end="\t\t")
            print()

        # Since it is undirected graph , adjacency Matrix will be symmetric
        for i in range(len(self.arr)):
            for j in range(i):
                print(self.arr[i][j], end="\t\t")
            print()

    def displayTransportHub(self):
        """
        Description: Loops over adjacency matrix to find the city with maximum number of trains passing through.
        """
        max_count = 0
        max_index = 0
        for i in range(len(self.arr)):
            tmp_list_trains = list()
            for j in range(len(self.arr)):
                if self.arr[i][j] != 0:
                    tmp_list_trains.extend(str(self.arr[i][j]).split(','))
            curr_count = len(self.unique(tmp_list_trains))
            if curr_count > max_count:
                max_index = i
                list_trains = self.unique(tmp_list_trains)
                max_count = curr_count

        f = open("outputPS22.txt", "a")
        f.write("--------Function displayTransportHub --------")
        f.write("\n")
        f.write("Main transport hub:" + self.uniq_cities[max_index].__str__())
        f.write("\n")
        f.write("Number of trains visited:" + len(list_trains).__str__())
        f.write("\n")
        f.write("List of Freight trains:")
        f.write("\n")
        for x in list_trains:
            f.write(x)
            f.write("\n")
        f.write("-----------------------------------------\n")
        f.write("\n")
        f.close()

    # function to check if small string is
    # there in big string
    def check(self, string, sub_str):
        if string.find(sub_str) == -1:
            return False
        else:
            return True

    def displayConnectedCities(self, train):
        """
        Description: Take train number as input and check it respective location in matrix.
        Once is location is identified then using the index 2 cities are identified. Since adjacency matrix
        are symmetric loop is done only on the lower triangular matrix.
        """
        list_of_connected_cities = list()
        for i in range(len(self.arr)):
            for j in range(i):
                if self.check(str(self.arr[i][j]), train):
                    list_of_connected_cities.append(self.uniq_cities[i])
                    list_of_connected_cities.append(self.uniq_cities[j])

        f = open("outputPS22.txt", "a")
        f.write("--------Function displayConnectedCities --------")
        f.write("\n")
        f.write("Freight train number:" + train)
        f.write("\n")
        f.write("Number of cities connected:" + len(self.unique(list_of_connected_cities)).__str__())
        f.write("\n")
        f.write("List of cities connected directly by " + train + ":")
        f.write("\n")
        for x in self.unique(list_of_connected_cities):
            f.write(x)
            f.write("\n")
        f.write("-----------------------------------------\n")
        f.write("\n")
        f.close()

    def displayDirectTrain(self, city_1, city_2):
        """
        Description: Take 2 city as input and find the their respective index and using their
        index checks in adjacency matrix at that specific location whether train is available or not.
        """
        package_can_be_sent = 'No, Apology no direct trains available on this route'
        if self.arr[self.uniq_cities.index(city_1)][self.uniq_cities.index(city_2)] != 0:
            package_can_be_sent = 'Yes, Package can be sent through ' + self.arr[self.uniq_cities.index(city_1)][
                self.uniq_cities.index(city_2)]
        f = open("outputPS22.txt", "a")
        f.write("--------Function displayDirectTrain --------")
        f.write("\n")
        f.write("City A:" + city_1.__str__())
        f.write("\n")
        f.write("City B:" + city_2.__str__())
        f.write("\n")
        f.write("Package can be sent directly:" + package_can_be_sent + "\n")
        f.write("-----------------------------------------\n")
        f.write("\n")
        f.close()

    def findServiceAvailable(self, city_1, city_2):
        package_can_be_sent = 'Yes'
        f = open("outputPS22.txt", "a")
        f.write("--------Function findServiceAvailable --------")
        f.write("\n")
        f.write("City A:" + city_1.__str__())
        f.write("\n")
        f.write("City B:" + city_2.__str__())
        f.write("\n")
        f.write("Package can be sent directly:" + package_can_be_sent + "\n")
        f.write("-----------------------------------------\n")
        f.write("\n")
        f.close()


## Class OnlineRailwayFreightBooking End

def main():
    """
    Description: main() function
    """
    inFile = "inputPS22.txt"  # name of input file
    promptFile = "promptsPS22.txt"  # name of the prompts file

    orfb = OnlineRailwayFreightBooking()  # creates an object of the OnlineRailwayFreightBooking class
    orfb.readApplications(inFile)  # populates the adjacency matrix.
    orfb.showAll()  # outputs all information to the output file
    orfb.populateAdjacencyMatrix() # populates the adjacency matrix.
    # orfb.printAdjacencyMatrix()

    file = open(promptFile, "r")
    prompts = file.readlines()  # store all the prompts into a list
    file.close()

    for prompt in prompts:
        if prompt.strip() == "searchTransportHub":  # condition to trigger searchTransportHub()
            orfb.displayTransportHub()

        elif "searchTrain" in prompt.strip():  # condition to trigger displayConnectedCities()
            inputs = prompt.strip().split(':')
            orfb.displayConnectedCities(inputs[1].strip())

        elif "searchCities" in prompt.strip():  # condition to trigger displayDirectTrain()
            inputs = prompt.strip().split(':')
            orfb.displayDirectTrain(inputs[1].strip(), inputs[2].strip())

        elif "ServiceAvailability" in prompt.strip():  # condition to trigger findServiceAvailable()
            inputs = prompt.strip().split(':')
            orfb.findServiceAvailable(inputs[1].strip(), inputs[2].strip())

    print("\n\nCompleted. Check outputPS22.txt in the current directory for output.\n\n")


if __name__ == "__main__":
    main()
