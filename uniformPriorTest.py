from examplePolicy import *
from snellenPolicy import *
from binaryBetaPolicy import *
from rootFindingPolicy import *
from exploreFit import *
from thompsonLossMin import *
import scipy.stats as stats
import numpy as np
import random

N_EXPERIMENTS = 200

EXP_DESCRIPTION = 'minLossThompson with expected loss based x_t'

'''
Size is in a range from 1 through 10
'''

FLOOR = (1. / 4.)
C = 0.8

outlog = open('logs/uniformLog2.csv', 'a+')

def main():
	Policy = ThompsonPolicy
	for paramIndex in range(20):
		paramValue = Policy.getParamValue(paramIndex)
		nMu, errorMu = bootstrapExperiments(paramValue, Policy)

		print(f'{paramValue}, {nMu}, {errorMu}')

def bootstrapExperiments(paramValue, Policy):
	ns = []
	errors = []
	for i in range(N_EXPERIMENTS):
		truthParams = sampleAPF()
		if i % 10 == 0: 
			print(i)
			outlog.flush()
		# print(f'True params: {truthParams}')
		n, error, prediction = runPatientTest(truthParams, paramValue, Policy)
		# print(f' => prediction: {prediction:.2f}')
		# print(f' => loss: {error:.2f}')
		# print(f' => n: {n}')
		ns.append(n)
		errors.append(error)
		k0 = truthParams[0]
		k1 = truthParams[1]
		outlog.write(f'{k0:.2f}, {k1:.2f}, {prediction:.2f}, {n}, {error}, {EXP_DESCRIPTION}\n')
		#print(i, truthParams, '=>', prediction)
	return np.mean(ns), np.mean(errors)

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
	# random.seed(0)
	# np.random.seed(seed=0)
	main()