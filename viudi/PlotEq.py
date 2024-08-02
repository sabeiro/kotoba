import numpy as np
import matplotlib.pyplot as plt
import pylab

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)


plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()

x = np.linspace(-15,15,100) # 100 linearly spaced numbers
y = np.sin(x)/x # computing the values of sin(x)/x

# compose plot
pylab.plot(x,y) # sin(x)/x
pylab.plot(x,y,'co') # same function with cyan dots
pylab.plot(x,2*y,x,3*y) # 2*sin(x)/x and 3*sin(x)/x
pylab.show() # show the plot

#numpy.polynomial.legendre.Legendre(coef, domain=None, window=None)[source]
