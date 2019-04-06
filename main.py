import math
import numpy as np
import matplotlib.pyplot as plt
import random

## simulation
s0 = 100.0		# initial price
x0 = 10000.0	# starting cash value
sigma = 0.01	# variance
M = 4000		# steps
Sim = 1000		# number of simulations
A = 0.5			# arrival intensity (lambda)
k = 1.5			# arrival intensity (lambda)

## parameters
Delta = 0.5
Give = 0.5

def simulate():
	s = -1
	while np.min(s) < 0:
		s = s0 + np.cumsum(sigma * np.random.standard_normal((M, Sim)), axis=0)
	x = x0 * np.ones((M, Sim))
	q = x0 / s0 * np.ones((M, Sim))
	for i in range(M-1):
		q_prime = (q[i] * s[i] - x[i]) / (q[i] * s[i] + x[i])
		ask_quote = np.maximum(np.multiply(s[i], 1 - q_prime * Give/100 + Delta/100), s[i])
		bid_quote = np.minimum(np.multiply(s[i], 1 - q_prime * Give/100 - Delta/100), s[i])
		ask_action = 1.0 * (np.random.random(Sim) < A * np.exp(- k * (ask_quote - s[i])))
		bid_action = 1.0 * (np.random.random(Sim) < A * np.exp(- k * (s[i] - bid_quote)))
		ask_action = np.minimum(ask_action, q[i])
		bid_action = np.minimum(bid_action, np.divide(x[i], bid_quote))
		q[i+1] = q[i] - ask_action + bid_action
		x[i+1] = x[i] + np.multiply(ask_action, ask_quote) - np.multiply(bid_action, bid_quote)
	start = x[0, :] + np.multiply(s[0, :], q[0, :])
	end = x[-1, :] + np.multiply(s[-1, :], q[-1, :])
	pnl = end - start
	roi = 100.0 * np.divide(pnl, start)
	print(np.min(roi), np.mean(roi), np.max(roi))

def main():
    simulate()

if __name__ == "__main__":
    main()
