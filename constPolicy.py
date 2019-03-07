import numpy as np
import math
import random
import numpy.random as ra
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm
'''
Model:
Floored exponential acuity response curve
Model slip probability
Gumbel prior distribution on vision

Policy:
Likelihood weighting to sample from the posterior of P(theta | evidence)
Thomspon sampling to chose the next value of K1
'''
class ConstPolicy:

	def setFloorProbability(self, newFloor):
		pass

	@staticmethod
	def getParamValue(paramIndex):
		return 0

	def __init__(self, nQuestions):
		pass

	def getNextSize(self):
		return 2

	def recordResponse(self, size, correct):
		pass

	def isDone(self):
		return True

	def getAnswer(self):
		return 1.3