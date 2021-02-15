class ItemValue:
    """Item Value DataClass"""

    def __init__(self, wt, val, ind):
        self.wt = wt
        self.val = val
        self.ind = ind
        self.cost = val // wt

    def __lt__(self, other):
        return self.cost < other.cost

# Greedy Approach
class MaxCalories:

    @staticmethod
    def getMaxValue(wt, val, capacity):
        """function to get maximum value """
        iVal = []
        for i in range(len(wt)):
            iVal.append(ItemValue(wt[i], val[i], i))

        # sorting items by value
        iVal.sort(reverse=True)

        print(iVal)

        totalValue = 0
        for i in iVal:
            curWt = int(i.wt)
            curVal = int(i.val)
            if capacity - curWt >= 0:
                capacity -= curWt
                totalValue += curVal
            else:
                fraction = capacity / curWt
                totalValue += curVal * fraction
                capacity = int(capacity - (curWt * fraction))
                break
        return totalValue


# Driver Code
if __name__ == "__main__":
    wt = [5,3,8,2,1,2]
    val = [110, 150, 144,120,35,44]
    capacity = 10

    # Function call
    maxValue = MaxCalories.getMaxValue(wt, val, capacity)
    print("Maximum value in  =", maxValue)
