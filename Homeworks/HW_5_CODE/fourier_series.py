import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

dx = 0.001
L = np.pi
x = L*np.arange(-1+dx, 1+dx, dx)
n = len(x)
nq = int(np.floor(n/4))
ub = np.pi/2

def fx(x, ub):
    if abs(x) < ub:
        return (np.pi/2)-abs(x)    
    else:
        return 0
    

f = np.array([fx(e,ub) for e in x])

fig, ax = plt.subplots()
ax.plot(x, f, '-', color='b', lw=2, label="Original hat function")
ax.set_xlabel("$x$")
ax.set_ylabel("$f(x)$")
plt.savefig("hat_function.png")

# Compute Forurier series\
name = "Accent"
cmap = get_cmap('tab10')
colors  = cmap.colors
ax.set_prop_cycle(color=colors)

A0 = np.sum(f * np.ones_like(x))*dx
fa = A0/2
A = np.zeros(20)
B = np.zeros(20)

Nmodes = 5

for k in range(Nmodes):
    A[k] = np.sum(f*np.cos(np.pi*(k+1)*x/L)) * dx
    B[k] = np.sum(f*np.sin(np.pi*(k+1)*x/L)) * dx
    fa= fa + A[k]*np.cos((k+1)*np.pi*x/L) +  B[k]*np.sin((k+1)*np.pi*x/L)
    lab="Modes_" + str(k) 
    ax.plot(x, fa, '-', label=lab)
    plt.legend()
    
plt.savefig("modes_function.png")

