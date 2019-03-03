from examplePolicy import *
from snellenPolicy import *
from binaryBetaPolicy import *
from rootFindingPolicy import *
from thompsonLossMin import *
from bayesianAcuityTest import *
from batWithPrior import *
from batPrecision import *
import scipy.stats as stats
import numpy as np
import random
import math

N_EXPERIMENTS = 100

EXP_DESCRIPTION = 'BAT precision SlipP = 0.05, Floor = 1/4'

'''
Size is in a range from 1 through 10
'''
C = 0.8
SLIP_P = 0.05

FLOORS = [
	1. / 4.,
	1. / 6.,
	1. / 10.,
	1. / 20.
]

outlog = open('logs/noisyTest_within10pp_2.csv', 'a+')

def main():
	floorP = FLOORS[0]
	Policy = BATPrecision
	nMu, errorMu = bootstrapExperiments(floorP, Policy)
	print(f'{floorP}, {nMu}, {errorMu}')

def bootstrapExperiments(floorP, Policy):
	ns = []
	errors = []
	for i in range(N_EXPERIMENTS):
		truthParams = sampleAPF()
		# truthParams = (3.,4.)
		if i % 10 == 0: 
			print(i)
			outlog.flush()
		# print(f'True params: {truthParams}')
		n, error, prediction = runPatientTest(truthParams, floorP, Policy)
		# print(f' => prediction: {prediction:.2f}')
		# print(f' => loss: {error:.2f}')
		# print(f' => n: {n}')
		ns.append(n)
		errors.append(error)
		k0 = truthParams[0]
		k1 = truthParams[1]
		outlog.write(f'{k0:.2f}, {k1:.2f}, {prediction:.2f}, {n}, {error}, {EXP_DESCRIPTION}\n')
		print(i, n, truthParams, '=>', prediction, '(' + str(error) + ')')
	return np.mean(ns), np.mean(errors)

def runPatientTest(truthParams, floorP, Policy):
	policy = Policy(0.1)
	policy.setFloorProbability(floorP)

	nDone = 0
	# print(truthParams)
	while not policy.isDone():
		size = policy.getNextSize()
		# print(size)
		answer = simulateResponse(truthParams, floorP,  size)
		policy.recordResponse(size, answer)
		nDone += 1

	x_hat = policy.getAnswer()
	x_star = truthParams[1]
	error = abs(x_star - x_hat)/x_star
	return nDone,error,x_hat

def simulateResponse(truthParams, floorP, size):
	if(size < 0): 
		return random.random() < floorP
	if(random.random() < SLIP_P):
		return random.random() < floorP
	k_0 = truthParams[0]
	k_1 = truthParams[1]

	pwr = (size - k_0) / (k_1 - k_0)
	# if you overflow here, then return floopP
	try:
		expP = 1 - pow(1-C, pwr)
		p = max(floorP, expP)
	except:
		p = floorP
	return random.random() < p
	
'''
This method samples k1 and k0.
log k1 ~ Gumbel(mu = )
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
	# random.seed(0)
	# np.random.seed(seed=0)
	main()