import math
import numpy as np
import matplotlib.pyplot as plt
import random

## simulation
s0 = 100.0		# initial price
x0 = 4000.0		# starting cash value
sigma = 0.05	# variance
M = 4000		# steps
Sim = 100		# number of simulations
A = 0.5			# arrival intensity (lambda)
k = 1.5			# arrival intensity (lambda)

## parameters
Delta = 0.1
Give = 0.5

def simulate():	
	s = -1
	while np.min(s) < 0:
		s = s0 + np.cumsum(sigma * np.random.standard_normal((M, Sim)), axis=0)
	x = x0 * np.ones((M, Sim))
	q = x0 / s0 * np.ones((M, Sim))
	ask_action = np.zeros(x.shape)
	bid_action = np.zeros(x.shape)
	ask_quotes = np.zeros(x.shape)
	bid_quotes = np.zeros(x.shape)
	for i in range(M-1):
		q_prime = (q[i] * s[i] - x[i]) / (q[i] * s[i] + x[i])
		ask_quotes[i] = np.maximum(np.multiply(s[i], 1 - q_prime * Give/100 + Delta/100), s[i])
		bid_quotes[i] = np.minimum(np.multiply(s[i], 1 - q_prime * Give/100 - Delta/100), s[i])
		ask_action[i] = 1.0 * (np.random.random(Sim) < A * np.exp(- k * (ask_quotes[i] - s[i])))
		bid_action[i] = 1.0 * (np.random.random(Sim) < A * np.exp(- k * (s[i] - bid_quotes[i])))
		ask_action[i] = np.minimum(ask_action[i], q[i])
		bid_action[i] = np.minimum(bid_action[i], np.divide(x[i], bid_quotes[i]))
		q[i+1] = q[i] - ask_action[i] + bid_action[i]
		x[i+1] = x[i] + np.multiply(ask_action[i], ask_quotes[i]) - np.multiply(bid_action[i], bid_quotes[i])
	start = x[0, :] + np.multiply(s[0, :], q[0, :])
	roi = np.zeros(x.shape)
	for i in range(1, M):
		end = x[i, :] + np.multiply(s[i, :], q[i, :])
		pnl = end - start
		roi[i, :] = 100.0 * np.divide(pnl, start)
	# chart: brownian market movements
	ylim = (np.min(s), np.max(s))
	f1 = plt.figure(figsize=(12, 8))
	ax1 = f1.add_subplot(111)
	ax1.plot(s, color='0.75')
	ax1.plot(s[:, 0], color='0')
	ax1.set_ylabel('Price ($)', fontsize=16, labelpad=16)
	ax1.set_xlabel('Steps', fontsize=16, labelpad=16)
	ax1.set_xlim(1, M)
	ax1.set_ylim(ylim)
	f1.savefig('paper/images/market_price.png', dpi=300, orientation='landscape', bbox_inches='tight')
	# chart: agent P&L's
	f2 = plt.figure(figsize=(12, 8))
	ax2 = f2.add_subplot(111)
	ax2.plot(roi, color='0.75')
	ax2.plot(np.mean(roi, axis=1), color='0')
	ax2.set_ylabel('ROI (%)', fontsize=16, labelpad=16)
	ax2.set_xlabel('Steps', fontsize=16, labelpad=16)
	ax2.set_xlim(1, M)
	f2.savefig('paper/images/roi.png', dpi=300, orientation='landscape', bbox_inches='tight')
	# chart: the orders placed by our agent
	f3 = plt.figure(figsize=(12, 8))
	ax3 = f3.add_subplot(111)
	ax3.plot(s[:, 0], color='0')
	asks = ask_action[:, 0] > 0
	bids = bid_action[:, 0] > 0
	ax3.scatter(np.arange(M)[asks], ask_quotes[asks, 0], 2, color='r')
	ax3.scatter(np.arange(M)[bids], bid_quotes[bids, 0], 2, color='g')
	ax3.set_ylabel('Price ($)', fontsize=16, labelpad=16)
	ax3.set_xlabel('Steps', fontsize=16, labelpad=16)
	ax3.set_xlim(1, M)
	f3.savefig('paper/images/orders.png', dpi=300, orientation='landscape', bbox_inches='tight')
	# chart: the inventory managed by a specific agent
	cash_value = x[:, 0]
	total_value = cash_value + s[:, 0] * q[:, 0]
	cash_value = np.divide(cash_value, total_value)
	f4 = plt.figure(figsize=(12, 8))
	ax4 = f4.add_subplot(111)
	ax4.fill_between(np.arange(M), 0, cash_value, facecolor='g', alpha=.7)
	ax4.fill_between(np.arange(M), cash_value, 1, facecolor='b', alpha=.7)
	ax4.set_ylabel('Inventory Value (%)', fontsize=16, labelpad=16)
	ax4.set_xlabel('Steps', fontsize=16, labelpad=16)
	ax4.set_xlim(1, M)
	ax4.set_ylim(0, 1)
	f4.savefig('paper/images/inventory.png', dpi=300, orientation='landscape', bbox_inches='tight')
	# show
	#plt.show()

def main():
    simulate()

if __name__ == "__main__":
    main()
