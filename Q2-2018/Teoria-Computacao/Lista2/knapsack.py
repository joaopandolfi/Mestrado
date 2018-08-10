
# Dynamic
def knapSack(W, wt, val, n):
    #Cria matriz de zeros
    K = [[0 for x in range(W+1)] for x in range(n+1)]
 
    #Executa algoritmo
    for i in range(n+1):
        for w in range(W+1):
            if (i==0 or w==0):
                K[i][w] = 0
            elif (wt[i-1] <= w):
                K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]],  K[i-1][w])
            else:
                K[i][w] = K[i-1][w]
 
    return K[n][W]
 
val = [20, 80, 150]
wt = [14, 26, 37]
W = 61
n = len(val)
print(knapSack(W, wt, val, n))
