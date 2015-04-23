import numpy as np;
####梯度下降法

##读入数据
###返回numpy数组格式的数据集X_dataSet和类标签Y_label
def load():
	X_dataSet=[];
	Y_label=[];
	fr = open('testSet.txt');
	for line in fr.readlines():
		lineArr = line.strip().split();
		X_dataSet.append([1.0,float(lineArr[0]),float(lineArr[1])]);
		Y_label.append([int(lineArr[2])]);
	return X_dataSet,Y_label;
	
## signmoid
def signmoid(z):  
	return 1.0/(1+np.exp(-z));
	
##进行梯度下降 
### data：特征矩阵，numpy数组格式
### label：类标签，numpy数组格式
#批梯度下降
def gradAscent(data,label):
	X_data = np.mat(data);
	X_label = np.mat(label);
	m,n = np.shape(X_data);
	alpha = -0.001 #步长
	w = np.ones((n,1));
	MAX = 500;
	for k in range(MAX):
		h = signmoid(X_data*w);
		w = w+alpha*(X_data.transpose()*(h-X_label));
	return w;

#随机梯度下降
#误差较大
def stocGradAscent(data,label) :
	m,n = np.shape(data);
	alpha = 0.001;
	w = np.ones(n);
	for i in range(m):
		h = signmoid(np.sum(data[i]*w));
		w= w+alpha*(label[i]-h)*data[i];
	w = np.mat(w);
	return w.transpose();
		

##可视化
### x特征矩阵，numpy数组格式
### y类标签，numpy数组格式
def plotFit(x,y,w):
	import matplotlib.pyplot as plt;
	x1=[]; y1=[]; #类别为1的点
	x2=[]; y2=[];
	w = w.getA();
	m = np.shape(x)[0];
	for i in range(m):
		if(y[i][0]==0) :
			x1.append(x[i][1]);
			y1.append(x[i][2]);
		else:
			x2.append(x[i][1]);
			y2.append(x[i][2]);
	plt.scatter(x1,y1,c='red');
	plt.scatter(x2,y2);
	z = np.mat(x)*np.mat(w);
	a=np.arange(-3.0,3.0,0.1);
	b=(-w[0]-w[1]*a)/w[2];
	plt.plot(a,b);
	plt.show();
	plt.figure(2);
# x,y=load();
# w = gradAscent(x,y);
# print(w);
# plotFit(x,y,w);
# w = stocGradAscent(x,y);
# print(w);
# plotFit(x,y,w);

##示例
def coli():
	frTrain = open('horseColicTraining.txt');
	frTest = open('horseColicTest.txt');
	X_dataSet=[];
	Y_labelSet=[];
	for line in frTrain.readlines():
		lineArr = line.strip().split();
		for i in range(21):
			lineMat=[1];
			lineMat.append(float(lineArr[i]));
		X_dataSet.append(lineMat);
		Y_labelSet.append([int(float(lineArr[21]))]);
	w = gradAscent(X_dataSet,Y_labelSet);
	error = 0;
	num = 0;
	for line in frTest.readlines():
		lineArr = line.strip().split();
		num += 1;
		for i in range(21):
			lineMat=[1];
			lineMat.append(int(float(lineArr[i])));
		lineMat = np.mat(lineMat);
		a = signmoid(lineMat*w);
		b=0;
		if(a[0]>0.5) :
			b = 1;
		else:
			b = 0;
		if(b != int(lineArr[21])):
			error+=1;
	print("error rate:",float(error)/num);

coli();
		


	
		
