import numpy as np 
import matplotlib.pyplot as plt 

def dudt(t,u):
	return alfa*u

alfa=1.0

dt=np.array([1.1, 1.5, 1.9])
A=2.

n=100000

h=dt/n

t = np.linspace(0.0,dt[1]*A,n)
u = np.zeros(n)


def RK4(t,u,h):
	u[0]=1.
	for i in range(1,n-1):
		k11 = h*dudt(t[i-1], u[i-1])
		k21 = h*dudt(t[i-1] + h/2, u[i-1] + k11/2)
		k31 = h*dudt(t[i-1] + h/2, u[i-1] + k21/2)
		k41 = h*dudt(t[i-1] + h, u[i-1] + k31)
		u[i] = u[i-1] + (k11 + 2*k21 + 2*k31 + k41)/6
	return u

def exacta(t):
	return np.exp(alfa*t)


plt.figure(figsize=(10,6))
plt.title("RK4")
for i in range(len(dt)):
	plt.plot(t, RK4(t,u,h[i]),label=dt[i])
plt.scatter(t,exacta(t),label='Exacta')

plt.grid()
plt.xlabel(r'$t$')
plt.ylabel(r'$u$')
plt.legend()
plt.show()