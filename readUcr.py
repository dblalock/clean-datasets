
import numpy as np
import os
from numpy import r_

import paths

UCR_DATASETS_DIR = paths.UCR

def readDataFile(path):
	D = np.genfromtxt(path)
	labels = D[:,0]
	X = D[:,1:]
	return (X, labels)


def nameFromDir(datasetDir):
	return os.path.basename(datasetDir)


def readUCRDataInDir(datasetDir, train):
	datasetName = nameFromDir(datasetDir)
	if train:
		fileName = datasetName + "_TRAIN"
	else:
		fileName = datasetName + "_TEST"
	filePath = os.path.join(datasetDir,fileName)
	return readDataFile(filePath)


def readUCRTrainData(datasetDir):
	return readUCRDataInDir(datasetDir, train=True)


def readUCRTestData(datasetDir):
	return readUCRDataInDir(datasetDir, train=False)


# combines train and test data
def readAllUCRData(ucrDatasetDir):
	X_train, Y_train = readUCRTrainData(ucrDatasetDir)
	X_test, Y_test = readUCRTestData(ucrDatasetDir)
	X = r_[X_train, X_test]
	Y = r_[Y_train, Y_test]
	return (X,Y)


def getAllUCRDatasetDirs():
	datasetsPath = os.path.expanduser(UCR_DATASETS_DIR)
	files = os.listdir(datasetsPath)
	for i in range(len(files)):
		files[i] = os.path.join(datasetsPath, files[i])
	dirs = filter(os.path.isdir, files)
	return dirs

if __name__ == '__main__':
	import sequence as seq

	# print out a table of basic stats for each dataset to verify
	# that everything is working
	nameLen = 22
	print("%s\tTrain\tTest\tLength\tClasses" % (" " * nameLen))
	for i, datasetDir in enumerate(getAllUCRDatasetDirs()):
		Xtrain, _ = readUCRTrainData(datasetDir)
		Xtest, Ytest = readUCRTestData(datasetDir)
		print('%22s:\t%d\t%d\t%d\t%d' % (nameFromDir(datasetDir),
			Xtrain.shape[0], Xtest.shape[0], Xtrain.shape[1],
			len(seq.uniqueElements(Ytest))))
