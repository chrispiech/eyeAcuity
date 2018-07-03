from examplePolicy import *
from snellenPolicy import *
from binaryBetaPolicy import *
from rootFindingPolicy import *
import scipy.stats as stats
import numpy as np
import random
import matplotlib.pyplot as plt

N_EXPERIMENTS = 100000

'''
Size is in a range from 1 through 10
'''

FLOOR = (1. / 4.)
C = 0.8

def main():
	k0s = []
	k1s = []
	for i in range(N_EXPERIMENTS):
		truthParams = sampleAPF()
		k0s.append(truthParams[0])
		k1s.append(truthParams[1])

	print('mu', np.mean(k1s))
	print('std', np.std(k1s))

	plt.hist2d(k0s, k1s)
	plt.colorbar()
	plt.show()

	plt.hist(k1s)
	plt.show()



def runPatientTest(truthParams, paramValue, Policy):
	#print(truthParams)
	policy = Policy(paramValue)

	nDone = 0
	while not policy.isDone():
		size = policy.getNextSize()
		answer = simulateResponse(truthParams, size)
		policy.recordResponse(size, answer)
		nDone += 1

	x_hat = policy.getAnswer()
	x_star = truthParams[1]
	error = abs(x_star - x_hat)/x_star
	return nDone,error,x_hat

def simulateResponse(truthParams, size):
	if(size < 0): 
		return random.random() < FLOOR
	k_0 = truthParams[0]
	k_1 = truthParams[1]

	pwr = (size - k_0) / (k_1 - k_0)
	expP = 1 - pow(1-C, pwr)
	p = max(FLOOR, expP)
	return random.random() < p
	

def sampleAPF():
	k_0 = stats.uniform.rvs(1, 6.5)
	maxRangeSize = min(3, 10 - k_0 - 0.5)
	rangeSize = stats.uniform.rvs(0.5, maxRangeSize)
	k_1 = k_0 + rangeSize
	return k_0, k_1

def frange(small, large, delta):
	values = []
	curr = small
	while curr <= large:
		values.append(curr)
		curr += delta
	return values


if __name__ == '__main__':
	main()