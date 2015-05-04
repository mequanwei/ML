from numpy import *;
from matplotlib.pyplot import *;
def loadData(filename):
	dataMat=[];
	fr = open(filename);
	for line in fr.readlines():
		cur = line.strip().split('\t');
		#python3 map返回iterators，需转换为list
		flt = list(map(float,cur));
		dataMat.append(flt);
	return dataMat;

def dist(a,b):
	return sqrt(sum(power(a-b,2)));

def randCent(dataSet,k):
	n = shape(dataSet)[1];
	cent = mat(zeros((k,n)));
	for j in range (n):
		minJ = min(dataSet[:,j]);
		#print(dataSet);
		rangeJ = float(max(dataSet[:,j])-minJ);
		cent[:,j] = minJ +rangeJ*random.rand(k,1);
	return cent;

def plotkMeans(dataSet,cent):
	scatter(dataSet[:,0],dataSet[:,1]);
	plot(cent[:,0],cent[:,1],'r+',);
	show();
	
def kMeans(dataSet,k):
	m=shape(dataSet)[0];
	clusterAss = mat(zeros((m,2)));
	cent =  randCent(dataSet,k);
	clusterChanged = True;
	while(clusterChanged):
		clusterChanged=False;
		for i in range(m):
			mindist = inf;minIndex=-1;	
			for j in range(k):
				distJ = dist(cent[j,:],dataSet[i,:]);
				if(distJ < mindist):
					mindist = distJ;
					minIndex = j;
			if (clusterAss[i,0]!=minIndex):
				clusterChanged=True;
			clusterAss[i,:]= minIndex,mindist**2
		for tcent in range(k):
			ptsIncluts = dataSet[nonzero(clusterAss[:,0].A==tcent)[0]];
			cent[tcent,:] = mean(ptsIncluts,axis=0);
	return cent,clusterAss;
	

def bikMeans(dataSet,k):
	m=shape(dataSet)[0];
	clusterAss = mat(zeros((m,2)));
	cent =  mean(dataSet,axis=0).tolist()[0];
	centlist=[cent];
	for j in range(m):
		clusterAss[j,1]=dist(mat(cent),dataSet[j,:])**2;
	while(len(centlist)<k):
		minsse = inf;
		for i in range(len(centlist)):
			ptsIncluts = dataSet[nonzero(clusterAss[:,0].A==i)[0],:];
			tcent,sselist = kMeans(ptsIncluts,2);
			seesplit = sum(sselist[:,1]);
			seenotsplit=sum(clusterAss[nonzero(clusterAss[:,0].A!=i)[0],1]);
			if((seesplit+seenotsplit)<minsse):
				minsse =seesplit+seenotsplit;
				index=i;
				splitcent=tcent;
				splitAss = sselist.copy();
		splitAss[nonzero(splitAss[:,0].A==1)[0],0]=len(centlist);
		splitAss[nonzero(splitAss[:,0].A==0)[0],0]=index;
		print ('the bestCentToSplit is: ',i)
		print ('the len of bestClustAss is: ', len(splitAss))
		centlist[index] = splitcent[0,:].tolist()[0];
		centlist.append(splitcent[1,:].tolist()[0]);
		clusterAss[nonzero(clusterAss[:,0].A == index)[0],:]= splitAss;
		#print(centlist);
	return centlist,clusterAss;
	
	
	
	
if __name__ =='__main__':
	dataSet = loadData("testSet2.txt");
	dataSet = mat(dataSet);
	cent,sse = bikMeans(dataSet,3);
	print(mat(cent));
	plotkMeans(dataSet,mat(cent));