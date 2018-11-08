import numpy as np
import matplotlib.pyplot as plt
import scipy.special

bipolar_sigmoid = lambda x: 2*scipy.special.expit(x)-1

x = np.linspace(-10,10,num=100)
y = bipolar_sigmoid(x)

plt.plot(x,y)

plt.show()