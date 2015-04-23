from numpy import  *;
import sys;
#测试数据
def load() :
	dataMat =  matrix([[1,2.1],
						 [2,1.1],
						 [1.3,1],
						 [1,1],
						 [2,1]]);
	classLabels = [1.0,1.0,-1.0,-1.0,1.0];
	return dataMat, classLabels;

#建立单层决策树(弱分类器)
def stumpClassify(x,y,w):
	x =  mat(x);
	y =  mat(y).T;
	m,n =  shape(x);
	g={};
	#步数为10
	step  = 10.0;
	ierr= 2.71828;#初始化错误率
	iarr=[]; #记录分类信息
	for i in range(n):
		max = x[:,i].max();
		min = x[:,i].min();
		#步长
		stepsize= float(max-min)/step;
		for j in range(-1,int(step)+1):
			for c in ['lt','gt']:
				val = min + j*stepsize;
				preArr= tryclassify(x[:,i],y,val,c);
				errArr =  mat(ones((m,1)));#记录错误的点
				errArr[preArr==y] = 0;
				werr = w.T*errArr;
				if(werr < ierr):
					ierr = werr;
					iarr=preArr.copy();
					g['dim']=i;
					g['val']=val;
					g['c']=c;	#记录分类器的信息
	return g,ierr,iarr; 

##决策树选择最佳划分
def tryclassify(x,y,val,c):
	m =  shape(x)[0];
	arr =  ones((m,1));
	if(c=='lt'):
		arr[x[:] <= val] = -1.0;
	else :
		arr[x[:] > val] = -1.0;
	return arr;

##x:m*n
##y:1*m	
def adaboostDS(x,y,num=40):
	x=mat(x);
	y=mat(y);
	weakClass=[];
	m =  shape(x)[0];
	w =  mat(ones((m,1))/m);
	arr = mat(zeros((m,1)));
	for i in range(num):
		g,werr,iarr = stumpClassify(x,y,w);
		alpha=float(0.5 * log((1-werr)/max(werr,1e-16)));
		g['alpha']=alpha;
		weakClass.append(g);
		n = -1*alpha*multiply(y.T,iarr);
		w = multiply(w, exp(n));
		w = w/w.sum();
		arr += alpha*iarr;  #分类结果
		# 统计误分点个数
		t=multiply(sign(arr)!=y.T,ones((m,1)));
		count = t.sum();
		errrate = count/m;
		#print("count:",count);
		if(count==0): break;
	return weakClass;
	

##加载数据
def loadSet(filename):
	f_num = len(open(filename).readline().split('\t'));
	fr = open(filename);
	dataMat = []; labelMat=[];
	for line in fr.readlines():
		arr=[];
		temp = line.split('\t');
		for i in range(f_num-1):
			arr.append(float(temp[i]));
		dataMat.append(arr);
		labelMat.append(float(temp[-1]));
	return dataMat,labelMat;

if __name__=='__main__':
	x,y = loadSet('horseColicTraining2.txt');
	xt,yt = loadSet('horseColicTest2.txt');
	xt = mat(xt)
	m = shape(xt)[0];
	g = adaboostDS(x,y,500);
	arr = zeros((m,1));
	for i in g:
		alpha = i['alpha'];
		dim = i['dim'];
		iarr = tryclassify(xt[:,dim],yt,i['val'],i['c']);
		arr += alpha*iarr
	t = sign(arr);
	count=0;
	for i in range(m):
		if(t[i]!=yt[i]) :
			count+=1;
	print('total:%s,error:%s,error rate:%s'%(m,count,float(count)/m));
	
	


	
		


	