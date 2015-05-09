from numpy import *;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn import cross_validation;
from sklearn.cross_validation import KFold;
def toint(arr):
	m =shape(arr)[0];
	for i in range(m):
		arr[i] = (int)(arr[i]);
	return arr;

def loadtrain(filename):
	fr = open(filename);
	alldata = fr.readlines();
	trainMat=[];
	labelMat=[];
	for line in alldata[1:]:
		arr = line.split(',');
		trainMat.append(toint(arr[1:]));
		labelMat.append((int)(arr[0]));
	trainMat = array(trainMat);
	labelMat = array(labelMat);
	fr.close();
	print('train data load success');
	return trainMat,labelMat;
	
def loadtest(filename):
	fr = open(filename);
	alldata = fr.readlines();
	testdata=[];
	for line in alldata[1:]:
		arr = line.split(',');
		testdata.append(toint(arr));
	fr.close();
	print('test data load success');
	return testdata;

##直接使用KNN	
##交叉验证观察效果
def knntest(trainMat,labelMat):
	m = shape(trainMat)[0];
	kf = KFold(m,n_folds=10);
	i = 0;
	for train,test in kf:
		xtrain = trainMat[train];
		ytrain = labelMat[train];
		xtest = trainMat[test]
		ytest = labelMat[test];
		neigh = KNeighborsClassifier();
		neigh.fit(xtrain,ravel(ytrain));
		error = 0;
		total= 0;
		mm = shape(xtest)[0]
		for j in range(mm):
			d = xtest[j];
			l = ytest[j];
			if(neigh.predict(d)!=l):
				error+=1;
			total+=1;
			i+=1;
		print(i,error,total,float(error)/total);
## 预测
def knn(x,y,t):
	neigh = KNeighborsClassifier();
	neigh.fit(x,y);
	fw = open('so.txt','w');
	for d in t:
		fw.write((str)(neigh.predict(d)[0]))
		fw.write('\n');
	fw.close();	

if __name__ =='__main__':
	x,y=loadtrain('train.txt');
	# knntest(x,y);
	t = loadtest('test.csv');
	knn(x,y,t);
	



		
	