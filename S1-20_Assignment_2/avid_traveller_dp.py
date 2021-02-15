
def maxCalories(W, wt, val, n):
    # Base Case
    if n == 0 or W == 0:
        return 0

    # If weight of the nth item is
    # more than maxCalories of capacity W,
    # then this item cannot be included
    # in the optimal solution
    if (wt[n - 1] >= W):
        return maxCalories(W, wt, val, n - 1)

    # return the maximum of two cases:
    # (1) nth item included
    # (2) not included
    else:
        return max(
            val[n - 1] + maxCalories(
                W - wt[n - 1], wt, val, n - 1),
            maxCalories(W, wt, val, n - 1))


# end of function maxCalories

# Driver Code
val = [22, 50, 18,60,35,22]
wt = [5,3,8,2,1,2]
W = 10
n = len(val)
print(maxCalories(W, wt, val, n))
