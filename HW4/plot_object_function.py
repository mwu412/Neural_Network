import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # projection='3d'

def func(x,y):
    return np.cos(0.7*x + 2.0*y) + 0.8*x*y

x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)

X , Y = np.meshgrid(x, y)

Z = func(X, Y)

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z)  # Axes3D.plot_surface(X, Y, Z, *args, **kwargs)
ax.plot_surface(X, Y, np.ones([100, 100]),color='r')  # Axes3D.plot_surface(X, Y, Z, *args, **kwargs)
ax.plot_surface(X, Y, -1*np.ones([100, 100]),color='r')  # Axes3D.plot_surface(X, Y, Z, *args, **kwargs)
ax.set_title('Object Function')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()

