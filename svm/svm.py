##SVM
import numpy as np;
def load(filename):
	fr = open(filename);
	data=[];label=[];
	for line in fr.readlines():
		lineArr = line.strip().split('\t');
		data.append([float(lineArr[0]),float(lineArr[1])]);
		label.append(float(lineArr[2]));
	return data,label;

#简化版选择alpha_j
def Salphaj(i,m):
	j = i;
	while(j==i):
		j = np.random.uniform(i,m);
	return int(j);

#规范化alpha_j
def clip(aj,H,L):
	if(aj>=H):
		aj=H;
	elif(aj<=L) :
		aj=L;
	return aj;
	
def antiKKT(alphai,ui,yi,toler,C):	
	
	if(yi*ui < -toler and alphai<C or yi*ui > toler and alphai>0 ):
		return True;
	return False;
	
def LH(alphai,alphaj,a,C):
	if(a):
		return max(0,alphai+alphaj-C),min(C,alphai+alphaj);
	else:
		return max(0,alphaj-alphai),min(C,C+alphaj-alphaj);

#X:数据 numpy数组格式
#Y:标签 numpy数组格式
#alpha 参数，numpy数组格式
#C：参数 实数
def SMO(X,Y,C,toler,max):
	X = np.mat(X);
	Y = np.mat(Y).T;
	m,n = np.shape(X);
	alpha = np.mat(np.zeros((m,1)));
	b=0;
	iter = 0;
	while(iter<max):
		change = 0;
		for i in range(m):
			xi=X[i,:]
			yi=Y[i,:]
			ui = np.multiply(alpha,Y).T*(X*xi.T)+b;
			ei = float(ui)-float(yi);
			alphai = alpha[i].copy();
			#alphai满足KKT就跳过，选择下一个alphai。否则，选择alphaj进行优化
			if(antiKKT(alphai,ei,yi,toler,C)):
				j = Salphaj(i,m);
				alphaj = alpha[j].copy();
				xj = X[j,:];
				yj= Y[j,:];
				uj = np.multiply(alpha,Y).T*(X*xj.T)+b;
				ej = float(uj)-float(yj);
				L,H = LH(alphai,alphaj,yi==yj,C);
				if L==H:  continue;
				eta = 2*(xi*xj.T)-(xi*xi.T)-(xj*xj.T);
				if eta >= 0: continue;
				alpha[j] = alphaj - yj*(ei-ej)/eta;
				alpha[j] = clip(alpha[j],H,L);
				if(abs(alpha[j]-alphaj)<0.00001):continue;
				alpha[i] = alphai+yi*yj*(alphaj-alpha[j]);
				b1=b-ei-yi*(alpha[i]-alphai)*(xi*xi.T)-yj*(alpha[j]-alphaj)*(xi*xj.T);
				b2=b-ej-yi*(alpha[i]-alphai)*(xi*xi.T)-yj*(alpha[j]-alphaj)*(xj*xj.T);
				if(alpha[i]>0 and alpha[i]<C):	b = b1;
				elif((alpha[j]>0 and alpha[j]<C)): b= b2;
				else:	b = (b1+b2)/2.0;
				change = change+1;
		if(change == 0) :iter += 1;
		else:iter=0;
	return alpha,b

##可视化
### x特征矩阵，numpy数组格式
### y类标签，numpy数组格式
def plotFit(x,y,alpha,b):
	import matplotlib.pyplot as plt;
	x1=[]; y1=[]; #类别为1的点
	x2=[]; y2=[];
	m = np.shape(x)[0];
	for i in range(m):
		if(y[i]==-1) :
			x1.append(x[i][0]);
			y1.append(x[i][1]);
		else:
			x2.append(x[i][0]);
			y2.append(x[i][1]);
	plt.scatter(x1,y1,c='red');
	plt.scatter(x2,y2);
	a=np.arange(2.0,8.0,0.1);
	x=np.mat(x);
	y=np.mat(y).T;
	w= sum(np.multiply(np.multiply(alpha,y),x));
	w0 =w[0,0];
	w1 = w[0,1];
	b0 = b;
	y = (-w0*a - b[0,0])/w1;
	plt.plot(a,y);
	plt.show();
	plt.figure(2);

data,label = load('testSet.txt');
alpha,b = SMO(data,label,0.6,0.001,40);
print(alpha[alpha>0]);
print(b)
plotFit(data,label,alpha,b)