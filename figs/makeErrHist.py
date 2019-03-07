import csv
import numpy as np
import math

FILE = '../logs/StAT-paper.csv'

# note that bins are in log space
def getBins():
	delta = 0.1
	return np.arange(-.3, 1.0 + delta, step=delta)

def getIndex(bins, v):
	if v < bins[0]:
		return 0
	if v > bins[-1]:
		raise Exception('value out of range: ', v, bins[-1])
	# who cares if it is slow...
	for i in range(len(bins) - 1):
		upper = bins[i + 1]
		if upper > v:
			return i

def main():
	reader = csv.reader(open(FILE))
	bins = getBins()

	# keep a list of errors for each bin
	buckets = []
	for i in range(len(bins)-1):
		buckets.append([])

	for line in reader:
		# for now we only care about k1 (not k0)
		k1 = float(line[1])
		err = float(line[4])

		# bins are in log space, so 
		logK1 = math.log(k1, 10)
		i = getIndex(bins, logK1)
		buckets[i].append(err)

	nSum = 0
	for bucket in buckets:
		n = len(bucket)
		aveErr = np.mean(bucket)
		std = np.std(bucket)
		stdErr = std / math.sqrt(n)
		print(n, aveErr, stdErr)
		nSum += n
	print(nSum)

if __name__ == '__main__':
	main()