from examplePolicy import *
from snellenPolicy import *
from binaryBetaPolicy import *
from rootFindingPolicy import *
from fract2 import Fract2
# from rootNeighbourFindingPolicy import *
# from rootMultipleSamplePolicy import *
# from exponentialFit2 import *
import scipy.stats as stats
import numpy as np
import random

import math

import matplotlib.pyplot as plt

N_EXPERIMENTS = 500

'''
Size is in a range from 1 through 10
'''
SLIP_P = 0.05
FLOOR = (1. / 4.)
C = 0.8

# If a policy takes more than this many iterations, we will stop early
MAX_N = 100

def main():

	testPolicies = {
		'snellen': SnellenPolicy,
		# 'rootFindng': RootFindingPolicy,
		'fract': Fract2
		# 'rootMultipleSample': RootMultipleSamplePolicy

		# 'rootNeighbourFindingPolicy': RootNeighbourFindingPolicy
		# 'binaryBeta': BinaryBetaPolicy

	}


	for name, policy in testPolicies.items():
		acc_vs_effort = testPolicy(policy, name=name)

		# unpack list of tuples into list of x and y values and plot
		plt.plot(*zip(*acc_vs_effort), label=name)


	print('\nDone!')
	plt.legend(loc='best')
	# plt.xlim(0, 35)
	plt.show()



def testPolicy(Policy, name=None):
	if name == None:
		name = str(Policy)

	print(f'\nTesting policy: {name}')
	print('='*20)
	ret = []
	for paramIndex in range(25):
		paramValue = Policy.getParamValue(paramIndex)
		nMu, errorMu = bootstrapExperiments(paramValue, Policy)

		ret.append((nMu, errorMu))

		if nMu > MAX_N:
			break

		print(f'[Param: {paramValue:g}] \tAverage n: {nMu:.2f}  Average err: {errorMu:.4e}')

	return ret


def bootstrapExperiments(paramValue, Policy):
	"""
	Evaluates policy on various floored exponential distributions.
	"""
	ns = []
	errors = []

	for i in range(N_EXPERIMENTS):
		# get new floored exponenetial distribution to test on
		truthParams = sampleAPF()
		# print(f'True params: {truthParams}')

		# test to learn truthParams
		n, error, prediction, hist = runPatientTest(truthParams, paramValue, Policy)

		# uncomment to visualise dynamics of search finding algorithm
		# visualise_hist(hist, truthParams)


		ns.append(n)
		errors.append(error)
		#print(i, truthParams, '=>', prediction)


	return np.mean(ns), np.mean(errors)

def runPatientTest(truthParams, paramValue, Policy):
	#print(truthParams)
	policy = Policy(paramValue)
	nDone = 0
	hist = []
	while not policy.isDone():
		# asks policy for next size to test and samples user's response at that size
		size = policy.getNextSize()
		answer = simulateResponse(truthParams, size)
		hist.append((size, answer))
		policy.recordResponse(size, answer)
		nDone += 1

	x_hat = policy.getAnswer()
	x_star = truthParams[1]
	error = abs(x_star - x_hat)/x_star

	return nDone,error,x_hat, hist

def visualise_hist(exp_hist, truthParams):
	"""
		- hist is list of (size, answer) pairs
	"""
	plt.ylim((0, 15))
	for hist in [exp_hist]:
		plt.plot(range(len(hist)), [h[0] for h in hist])
		plt.plot(range(len(hist)), [truthParams[1] for _ in hist])

	plt.show()



# def simulateResponse(truthParams, size):
# 	# random guessing
# 	if(size < 0):
# 		return random.random() < FLOOR


# 	k_0 = truthParams[0]
# 	k_1 = truthParams[1]

# 	pwr = (size - k_0) / (k_1 - k_0)
# 	expP = 1 - pow(1-C, pwr)
# 	p = max(FLOOR, expP)
# 	return random.random() < p


def simulateResponse(truthParams, size):
	if(size < 0):
		return random.random() < FLOOR

	if(random.random() < SLIP_P):
		return random.random() < FLOOR

	k_0 = truthParams[0]
	k_1 = truthParams[1]

	pwr = (size - k_0) / (k_1 - k_0)
	# if you overflow here, then return floopP
	try:
		expP = 1 - pow(1-C, pwr)
		p = max(FLOOR, expP)
	except:
		p = FLOOR
	return random.random() < p


'''
This method samples k1 and k0.
log k1 ~ Gumbel(mu = 0.3, beta = 0.5). Must be in the range
-.3 to 1.
'''
def sampleAPF():
	logK1 = rejectSampleGumbel(0.3, 0.5, -.3, 1.0)
	k1 = math.pow(10, logK1)
	locK0 = math.log(0.7 * k1, 10)
	logK0 = rejectSampleGumbel(locK0, 0.05, -10.0, logK1)
	k0 = math.pow(10, logK0)
	return k0, k1

	# maxRangeSize = min(3, 10 - k_0 - 0.5)
	# rangeSize = stats.uniform.rvs(0.5, maxRangeSize)
	# k_1 = k_0 + rangeSize
	# return k_0, k_1

def rejectSampleGumbel(loc, scale, minV, maxV):
	while True:
		x = stats.gumbel_r.rvs(loc, scale)
		if x > minV and x < maxV:
			return x

def frange(small, large, delta):
	values = []
	curr = small
	while curr <= large:
		values.append(curr)
		curr += delta
	return values




if __name__ == '__main__':
	main()