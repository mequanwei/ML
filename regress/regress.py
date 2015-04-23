##回归
from numpy import *;
from matplotlib.pyplot import *;

#X:m*n
#Y:m*1
def regress(x,y):
	xtx = x.T*x;
	#检验行列式的值是否为0
	if(linalg.det(xtx)==0.0):
		return;
	w = xtx.I*x.T*y;
	return w;

def loadData(filename):
	x=[];
	y=[];
	f_num = len(open(filename).readline().split('\t'));
	f = open(filename);
	for line in f.readlines():
		arr = [];
		temp = line.strip().split('\t');
		for i in range(f_num-1):	
			arr.append(float(temp[i]));
		x.append(arr);
		y.append(float(temp[-1]));
	return x,y;

#局部加权回归
def wregress(xi,x,y,k):
	m = shape(x)[0];
	g_w = mat(eye(m));
	for i in range(m):
		#高斯核
		diff = xi-x[i,:];
		g_w[i,i] = exp(diff*diff.T/(-2*k*k));
	xtx = x.T*(g_w*x);
	#检验行列式的值是否为0
	if(linalg.det(xtx)==0.0):
		return;
	w = xtx.I*(x.T*(g_w*y));
	return xi*w;	
	
if __name__ =='__main__':
	x,y=loadData('ex0.txt');
	x=mat(x);
	y=mat(y).T;
	#普通线性回归
	w=regress(x,y);
	print(w); 
	scatter(x[:,1],y);
	# 不排序画出来的图是错的
	t = x.copy()
	t.sort(0);
	y_hat = t*w;
	plot(t[:,1],y_hat,'red');
	#局部加权回归
	y_hat1=zeros(shape(x)[0]);
	for i in range(shape(x)[0]):
		y_hat1[i] = wregress(x[i,:],x,y,0.005);
	# 对x，y排序
	srtInt = x[:,1].argsort(0);
	xsort = x[srtInt][:,0,:];
	plot(xsort[:,1],y_hat1[srtInt],'black');
	show();


