import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm

ALGO = 'minLossThompson2' # fastThompson0 logThompson1

def main():
	reader = csv.reader(open('logs/uniformLog2.csv'))

	axis = getAxis()
	axisN = len(axis)
	total = np.zeros((axisN,axisN))
	count = np.zeros((axisN,axisN))

	marginal = np.zeros(axisN)
	marginalCount = np.zeros(axisN)

	losses = []

	for row in reader:
		k0 = float(row[0])
		k1 = float(row[1])
		pred = float(row[2])
		n = int(row[3])
		loss = float(row[4])
		algo = row[5].strip()

		if algo != ALGO: 
			continue
		#if n != 20: continue


		# matrix of k0, k1
		r = getIndex(k0)
		c = getIndex(k1)
		count[r][c] += 1
		total[r][c] += loss

		# marginal for k1
		marginal[c] += loss
		marginalCount[c] += 1

		# all losses
		losses.append(loss)
	print(np.mean(losses), len(losses))

	average = np.zeros(axisN)
	for i in range(axisN):
		if marginalCount[i] >= 5:
			average[i] = marginal[i] / marginalCount[i]
		else:
			average[i] = float('nan')
	figure = plt.figure(1)
	plt.plot(axis, average)
	figure.show()
	input()


	# average = np.zeros((axisN,axisN))
	# for i in range(axisN):
	# 	for j in range(axisN):
			
	# 		if count[i][j] <= 5:
	# 			average[i][j] = float('nan')
	# 		else:
	# 			average[i][j] = total[i][j] / count[i][j]
	# print np.mean(loss)
	# showMatrix(average)

def showMatrix(matrix):
	figure = plt.figure(1)
	plt.imshow(matrix, cmap=cm.coolwarm, interpolation='none');
	plt.colorbar()
	ax = plt.gca();

	axis = getAxis()

	axisLocations = []
	axisLabels = []
	for i in range(len(axis)):
		if i % 3 == 0:
			axisLocations.append(i)
			l = "{:.1f}".format(axis[i])
			axisLabels.append(l)
	ax.set_xticks(axisLocations)
	ax.set_xticklabels(axisLabels);
	ax.set_yticks(axisLocations)
	ax.set_yticklabels(axisLabels);
	figure.show()
	input()


def getIndex(value):
	axis = getAxis()
	for i in range(len(axis)-1):
		lower = axis[i]
		upper = axis[i+1]
		if value >= lower and value <= upper:
			return i
	raise Exception('out of bounds value: ' + str(value))

def getAxis():
	delta = .5
	return np.arange(1.0, 10. + delta, delta)

if __name__ == '__main__':
	main()
