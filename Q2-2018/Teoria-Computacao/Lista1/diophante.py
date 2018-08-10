# --------------------------------------
# João Carlos Pandolfi Santana (c) 2018
# --------------------------------------

# ----- FUNCOES ------

# Funcao para calcular o MDC
def euclides(a, b):
	r = a;
	r1 = b;
	u = 1;
	v = 0;
	u1 = 0;
	v1 = 1;
    
	while (r1 != 0):
		q = int(r / r1); # pega apenas a parte inteira
		rs = r;
		us = u;
		vs = v;
		r = r1;
		u = u1;
		v = v1;
		r1 = rs - q *r1;
		u1 = us - q*u;
		v1 = vs - q*v1;
	

	return [r, u, v]; # tais que a*u + b*v = r et r = pgcd (a, b)


# ----- Codigo ------

# -- Lendo dados
print("Resolvedor de equacoes diophantinas lineares")
print("Formato: Ax + By = C")
a = int(input("A = "))
b = int(input("B = "))
c = int(input("C = "))


# ---- Calculando GDC ----
gdcAB , n, m = euclides(a,b)

# --- Encontrando uma solucao ---
x0 = (n*c)/gdcAB
y0 = (m*c)/gdcAB

print("\nUma solucao pelo metodo comum")
print("x="+str(x0))
print("y="+str(y0))

# --- Econtrando todas as possiveis solucoes para LDE ---

xn = ((n*c)/gdcAB)
xnk = (b/gdcAB)

yn = ((m*c)/gdcAB)
ynk = (a/gdcAB)

print("\nSolucao usando o metodo do LDE")
print("x = "+str(xn)+" + "+str(xnk)+ "*k")
print("y = "+str(yn)+" - "+str(ynk)+ "*k")