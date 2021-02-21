class ItemValue:
    """Item Value DataClass"""

    def __init__(self, wt, val,fraction, ind):
        self.wt = wt
        self.val = val
        self.ind = ind
        self.fraction = fraction
        self.cost = val // wt

    def __lt__(self, other):
        return self.cost < other.cost

# Greedy Approach
class MaxCalories:

    @staticmethod
    def getMaxValue(wt, val, capacity,iVal):
        """function to get maximum value """
        # iVal = []


        # sorting items by value
        iVal.sort(reverse=True)

        for i in iVal:
            # iVal.append(ItemValue(wt[i], val[i], i))
            print(i.wt)
        list_of_food_items = []
        totalValue = 0
        for i in iVal:
            list_of_food_items.append(i)
            curWt = int(i.wt)
            curVal = int(i.val)
            if capacity - curWt >= 0:
                capacity -= curWt
                totalValue += curVal
            else:
                fraction = capacity / curWt
                totalValue += curVal * fraction
                capacity = int(capacity - (curWt * fraction))
                i.fraction = fraction
                break
        return list_of_food_items


# Driver Code
# if __name__ == "__main__":
#     wt = [5,3,8,2,1,2]
#     val = [110, 150, 144,120,35,44]
#     capacity = 10
#
#     # Function call
#     maxValue = MaxCalories.getMaxValue(wt, val, capacity)
#     print("Maximum value in  =", maxValue)

# Driver Code
if __name__ == "__main__":
    wt = []
    val = []
    capacity = int
    number_of_items = int
    list_of_food_items = []

    inputfile = "inputPS1.txt"

    file = open(inputfile, 'r')
    lines = file.readlines()
    file.close()
    maxValue = 0
    iVal = []
    for x in lines:
        if "Food Items" in x.strip():
            number_of_items = x.strip().split(':')[1].strip()

        elif "Maximum Bag Weight" in x.strip():
            capacity = int(x.strip().split(':')[1].strip())
        else:
            input = x.split("/")
            iVal.append(ItemValue(int(input[1].strip()), int(input[1].strip()) * int(input[2].strip()),0,input[0]))

    # Function call
    maxValue = MaxCalories.getMaxValue(wt, val, capacity,iVal)
    totalcalories = 0
    file = open("outputPS22.txt", "a")
    file.write("Total Calories: " +str(totalcalories)+ "\n\n")
    file.write("Food Item selection Ratio:\n")
    for x in maxValue:
        print(str(x.wt) + " " + str(x.val) + " " + str(x.fraction))
        val = x.fraction if x.fraction > 0 else x.wt
        file.write(x.ind + " : " + str(val) + "\n")
    file.close()
    print("Maximum value in  =", totalcalories)