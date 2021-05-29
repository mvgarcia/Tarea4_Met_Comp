import numpy as np
import matplotlib.pyplot as plt

print("-------------------Punto 2 parte 1 & 2-----------------------")
def f1(t,u,q):
	return u**q

n=100000
t0=0.
tf=-10.
h=(tf-t0)/n
print("Paso en Euler")
print("h=",h)
t = np.linspace(t0,tf,n)
u=np.zeros(n)

q=np.linspace(2.0,10.0,9)
# Implementar el metodo

def GetEuler(u,t,h,q):
	u[0]=1
	for i in range(1,n):
		u[i] = u[i-1] + f1(t[i-1],u[i-1],q)*h
	return u



plt.figure(figsize=(10,6))
plt.title("Euler")
for i in q:
	plt.plot(t, GetEuler(u,t,h,i), label=i)
plt.grid()
plt.xlabel(r'$t$')
plt.ylabel(r'$u$')
plt.legend()







print("-------------------Punto 2 parte 3-----------------------")

def y1(x):
	return x**(-2)

def dy2dx(x,y2):
	return x*(y2)**2

n=10000
x0=np.sqrt(2.0)
xf=10.
h=(xf-x0)/n
print("Paso en Euler")
print("h=",h)
x = np.linspace(x0,xf,n)
y2=np.zeros(n)
y=np.zeros(n)
q=np.linspace(2.0,10.0,9)

def GetEuler(y1,y2,x,h):
	y[0]=0.0
	for i in range(1,n):
		y2[i] = y2[i-1] + dy2dx(x[i-1],y2[i-1] )*h
		y[i] = y2[i-1] + y1(x[i-1])
	return y

plt.figure(figsize=(10,6))
plt.title("Euler Riccati dif. Eq.")
plt.plot(x, GetEuler(y1,y2,x,h))
plt.grid()
plt.xlabel(r'$x$')
plt.ylabel(r'$y=y_1 + y_2$')
plt.show()


