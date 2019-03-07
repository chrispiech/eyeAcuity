from examplePolicy import *
from snellenPolicy import *
from binaryBetaPolicy import *
from constPolicy import *
from rootFindingPolicy import *
from thompsonLossMin import *
from bayesianAcuityTest import *
from batWithPrior import *
from batCalibration import *
from fract2 import *
import scipy.stats as stats
import numpy as np
import random
import math
import json

N_EXPERIMENTS = 50

EXP_DESCRIPTION = 'Standard StAT for paper result. SlipP = 0.05, Floor = 1/4'
SAVE = True

'''
Size is in a range from 1 through 10
'''
C = 0.8
SLIP_P = 0.05

FLOOR = 1. / 4.
PRECISION = 0.1

outlog = open('logs/StAT-fake.csv', 'a+')
outlog_final = open('logs/StAT-calib.txt', 'w')

def main():
	Policy = BATCalibration

	# for paramIndex in range(20):
	# paramValue = Policy.getParamValue(paramIndex)
	nMu, errorMu, calib_info = bootstrapExperiments(Policy)


	print(f'===> {nMu}, {errorMu}')

	yTrue, yProb = list(zip(*calib_info))
	yTrue = list(yTrue)
	yProb = list(yProb)

	out = {
		'yTrue': yTrue,
		'yProb': yProb
	}

	print(out)

	json.dump(out, outlog_final)



def bootstrapExperiments(Policy):
	ns = []
	errors = []
	calib_info = []
	for i in range(N_EXPERIMENTS):
		truthParams = sampleAPF()
		# truthParams = (3.,4.)
		if i % 10 == 0:
			print(i)
			if SAVE: outlog.flush()
		# print(f'True params: {truthParams}')
		n, error, prediction, prob = runPatientTest(truthParams, Policy)

		in_err_range = int(error <= PRECISION)

		calib_info.append((in_err_range, prob))

		# if error <= PRECISION:
			# num_in_range += 1
		# else:
			# print(f'out of error: err={error}, precisions={PRECISION}, confidence={confidence}')
		# print(f' => prediction: {prediction:.2f}')
		# print(f' => loss: {error:.2f}')
		# print(f' => n: {n}')
		ns.append(n)
		errors.append(error)
		k0 = truthParams[0]
		k1 = truthParams[1]
		if SAVE: outlog.write(f'{k0:.2f}, {k1:.2f}, {prediction:.2f}, {n}, {error}, {EXP_DESCRIPTION}\n')

		print(i, n, truthParams, '=>', prediction, '(' + str(error) + ')')

	return np.mean(ns), np.mean(errors), calib_info

def runPatientTest(truthParams, Policy):
	priorK1 = truthParams[1]
	policy = Policy(PRECISION)
	# policy.setFloorProbability(FLOOR)

	nDone = 0
	# print(truthParams)
	while not policy.isDone():
		size = policy.getNextSize()
		# print(size)
		answer = simulateResponse(truthParams,  size)
		policy.recordResponse(size, answer)
		nDone += 1

	# x_hat = policy.getAnswer()
	x_hat, prob = policy.getBestAnswerSoFar()


	x_star = truthParams[1]
	error = abs(x_star - x_hat)/x_star
	return nDone, error, x_hat, prob

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
	# random.seed(0)
	# np.random.seed(seed=0)
	main()