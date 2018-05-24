from examplePolicy import *
import scipy.stats as stats
import numpy as np
import random

N_EXPERIMENTS = 100

'''
Size is in a range from 1 through 10
'''

FLOOR = 0.25
C = 0.8


def main():
	for alpha in frange(0., 1., 0.1):
		nMu, errorMu = bootstrapExperiments(alpha, ExamplePolicy)
		print(f'{alpha: .{4}}, {nMu}, {errorMu}')

def bootstrapExperiments(alpha, Policy):
	ns = []
	errors = []
	for i in range(N_EXPERIMENTS):
		truthParams = sampleAPF()
		n, error = runPatientTest(truthParams, alpha, Policy)
		ns.append(n)
		errors.append(error)
	return np.mean(ns), np.mean(errors)

def runPatientTest(truthParams, alpha, Policy):
	policy = Policy(alpha)
	nDone = 0
	while not policy.isDone():
		size = policy.getNextSize()
		answer = simulateResponse(truthParams, size)
		policy.recordResponse(size, answer)
		nDone += 1

	x_hat = policy.getAnswer()
	x_star = truthParams[1]
	error = pow((x_star - x_hat), 2)
	return nDone,error

def simulateResponse(truthParams, size):
	k_0 = truthParams[0]
	k_1 = truthParams[1]

	pwr = (size - k_0) / (k_1 - k_0)
	expP = 1 - pow(1-C, pwr)
	p = max(FLOOR, expP)
	return random.random() < p
	

def sampleAPF():
	b = stats.uniform.rvs(1, 10)
	rangeSize = stats.uniform.rvs(0.5, 5)
	k80 = b + rangeSize
	return b, k80

def frange(small, large, delta):
	values = []
	curr = small
	while curr <= large:
		values.append(curr)
		curr += delta
	return values


if __name__ == '__main__':
	main()