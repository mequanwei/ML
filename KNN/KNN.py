import numpy as np;
import matplotlib.pyplot as plt;
import os;
def createDataSet():
	group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]]);
	labels = ['A','A','B','B'];
	return group,labels;

group,labels = createDataSet();

def classify0(inX,dataSet,labels,k):
	diffMat = dataSet-np.tile(inX,(dataSet.shape[0],1));
	diffMat = diffMat ** 2;
	dist = diffMat.sum(axis=1) ** 0.5;
	sortedIndex = dist.argsort();
	count= {};
	for i in range(k):
		item = labels[sortedIndex[i]];
		count[item] = count.get(item,0) + 1;
	result = sorted(count.iteritems(),key=lambda d:d[1],reverse=True);
	return result[0][0];

# group, labels = createDataSet();
# c = classify0([1,2],group,labels,3);
# print(c);


def autoNorm(dataSet):
	min = dataSet.min(axis=0);
	max = dataSet.max(axis=0);
	row = dataSet.shape[0];
	dataSet = (dataSet-np.tile(min,(row,1)))/np.tile((max-min),(row,1));
	return dataSet;


def file2matrix(filename):
	fr=open(filename);
	arrayOLines = fr.readlines();
	numOfLines = len(arrayOLines);
	returnMat = np.zeros((numOfLines,3));
	classLabel=np.zeros((numOfLines,1));
	index=0;
	for line in arrayOLines:
		line = line.strip();
		lineFormLine =line.split('\t');
		returnMat[index,:] = lineFormLine[0:3];
		classLabel[index] = int(lineFormLine[-1]);
		index +=1;
	
	return returnMat,classLabel[:,0];

# dataSet,labelSet = file2matrix('datingTestSet2.txt');
# print(dataSet);
# print(autoNorm(dataSet));

def datingClassTest():
	hoRate=0.1;
	k=3;
	dataSet,labelSet = file2matrix('datingTestSet2.txt');
	dataSet = autoNorm(dataSet);
	row = int((1-hoRate)*dataSet.shape[0]);
	testSet = dataSet[row:,];
	testLabelSet = labelSet[row:,];
	error = 0;
	for item,rLabel in zip(testSet,testLabelSet):
		label = classify0(item,testSet,testLabelSet,k);
		if (rLabel != label) :
			error+=1;
	print("the total error rate is: %f",error/float(row));

#datingClassTest();

def img2vec(filename):
	vec = np.zeros((1,1024));
	fr = open(filename);
	for i in range(32):
		line = fr.readline();
		for j in range(32):
			vec[0,i*32+j]=int(line[j]);
	return vec;

def hwClassTest():
	hwLabel=[];
	trainingFileList = os.listdir("trainingDigits");
	m = len(trainingFileList);
	dataSet = np.zeros((m,1024));
	for i in range(m):
		digName = trainingFileList[i];
		label = int((digName.split('.')[0]).split('_')[0]);
		hwLabel.append(label);
		dataSet[i,:] = img2vec('trainingDigits/%s' % digName);
	testFileList = os.listdir('testDigits');
	error = 0;
	mTest = len(testFileList);
	for i in range(mTest):
		digName = testFileList[i];
		label = int((digName.split('.')[0]).split('_')[0]);
		testVec = img2vec('testDigits/%s' % digName);
		rLabel = classify0(testVec,dataSet,hwLabel,5);
		if(rLabel != label) :
			error+=1;
	print('error rate is %f' %(error/float(mTest)));
	
hwClassTest();
	
		

















