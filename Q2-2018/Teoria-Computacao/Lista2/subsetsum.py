def isSubsetSum(set, n, sum):
    subset = [[False for x in range(sum+1)] for x in range(n+1)]
  
    # Setando os primeiros valores como True
    for i in range(n+1):
        subset[i][0] = True;
  
    # Preenchendo os subsets
    for i in range(1,n+1):
        for j in range(1,sum+1):
            if(j<set[i-1]):
                subset[i][j] = subset[i-1][j];
            if (j >= set[i-1]):
                subset[i][j] = (subset[i-1][j] or subset[i - 1][j-set[i-1]])

    return subset[n][sum];


set = [3, 34, 4, 12, 5, 2]
sum = 9
n = len(set)-1
print(isSubsetSum(set, n, sum))