
# Converte SAT para 3sSAT
def satTo3Sat(A):

    tTotal = len(A)
    r= []
    for c in A:
        tam = len(c)

        if(tam == 1): # caso 1
            r.append([c[0],c[0],c[0]])
        elif(tam == 2): # caso 2
            r.append([c[0],c[1],c[1]])
        elif(tam == 3): #caso 3
            r.append([c])
        elif(tam > 3): #Quando tem que splitar

            cn = []
            ag = 0   
            cn.append([c[0],c[1], 'Y'+str(ag)])

            if(tam == 4):
                cn.append(['-Y'+str(ag),c[2],c[3]])

            elif(tam == 5):
                cn.append(['-Y'+str(ag),c[2],'Y'+str(ag+1)])
                ag+=1
                cn.append(['-Y'+str(ag),c[3],c[4]])

            else:
                for k in range(2,tam-2):
                    cn.append(['-Y'+str(ag),c[k],'Y'+str(ag+1)])
                    ag+=1
                cn.append(['-Y'+str(ag),c[3],c[4]])

            r.append(cn)
    return r

A = [[1,2,3,4],[3,-3,-5,5,6],[1,2,3,4,-5,-6,7]]

print(satTo3Sat(A))


# ================== KNAPSTACK
def KNAPSACK(S, peso, custo, n):

    dp = [[0 for x in range(S+1)] for x in range(n+1)]
    
    # Tabela dp[][]        
    for i in range(n+1):
        for s in range(S+1):
            if i==0 or s==0:
                dp[i][s] = 0
            elif peso[i-1] <= s:
                dp[i][s] = max(custo[i-1] + dp[i-1][s-peso[i-1]],  dp[i-1][s])
            else:
                dp[i][s] = dp[i-1][s] 

    return dp[n][S]


custo =  [3,4,6,10]
peso = [2,3,6,4]
S = 10
n = len(custo)
print(KNAPSACK(S, peso, custo, n))



# 3SAT to Clique

def Conv3satToClique(A):
    tam = len(A)
    G = {}
    for C in range(tam):
        curr = A.pop()

        for l in curr: #Cada elemento
            G[l]= []
            for cx in A: # Cada C
                for lx in cx: # Cada elemento
                    if(lx != (-1*l)):
                        G[l].append(lx) 

        A.insert(0,curr)
    return G

A = [[1,2,3],[-1,2,-1],[-1,1,1]]
print(Conv3satToClique(A))
