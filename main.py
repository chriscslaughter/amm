import math
import numpy as np
import matplotlib.pyplot as plt
import random

s0 = 100.0		# initial price
sigma = 2.0		# variance
M = 200			# steps
Sim = 1000		# number of simulations
A = 140			# arrival intensity (lambda)
k = 1.5			# arrival intensity (lambda)

def simulate():
	s = s0 + np.cumsum(sigma * np.random.standard_normal((Sim, M)), axis=1)
	f1 = plt.figure(figsize=(10, 8))
	ax1 = f1.add_subplot(111)
	ax1.plot(s[0])
	plt.show()
	for i in range(Sim):
		pass
		#print("Performing simulation {:4}/{:4} ..".format(i+1, Sim))

def main():
    simulate()

if __name__ == "__main__":
    main()
