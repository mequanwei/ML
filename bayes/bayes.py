import numpy as np;
def loadDataSet():
	postingList=[['my','dog','has','flea',\
					'problems','help','please'],
					['maybe','not','take','him',\
					'to','dog','park','stupid'],
					['my','dalmation','is','so','cute',\
					'I','love','him'],
					['stop','posting','stupid','worthless','grabage'],
					['mr','licks','ate','my','steak','how',\
					'to','stop','him'],
					['quit','buying','worthless','dog','food','stupid']];
	classVec=[0,1,0,1,0,1];
	return postingList,classVec;

def createVocList(dataSet):
	vocSet = set([]);
	for doc in dataSet:
		vocSet= vocSet|set(doc);
	return list(vocSet);
	
def setOfWord2Vec(vocList,inputSet):
	returnVec=[0]*len(vocList);
	for word in inputSet:
		if word in vocList:
			returnVec[vocList.index(word)] = 1;
		else:
			print("the word %s is not in my vocSet" %word);
	return returnVec;

# lists,classes = loadDataSet();			
# voac = createVocList(lists);
# print(voac)
# print(setOfWord2Vec(voac,lists[0]));

def trainNB0(trainSet,trainClass):
	numTrainData = len(trainSet);
	numWords = len(trainSet[0]);
	pAb = sum(trainClass)/float(numTrainData);
	p0Num = np.ones(numWords);
	p1Num = np.ones(numWords);
	p0Denom =2.0;
	p1Denom =2.0;
	for i in range(numTrainData):
			if(trainClass[i] == 1):
				p1Num += trainSet[i];
				p1Denom += sum(trainSet[i]);
			else:
				p0Num += trainSet[i];
				p0Denom += sum(trainSet[i]);
	p1Vect = np.log(p1Num/p1Denom);
	p0Vect = np.log(p0Num/p0Denom);
	return p0Vect,p1Vect,pAb;

# lists,classes = loadDataSet();			
# voa = createVocList(lists);
# trainSet = [];
# for data in lists:
	# trainSet.append(setOfWord2Vec(voa,data));
# a,b,c = trainNB0(trainSet,classes);
# print(a,b,c);
	
def classifyNB(v,p0Vec,p1Vec,pClass1):
	p0 = sum(v*p0Vec)+np.log(pClass1);
	p1 = sum(v*p1Vec)+np.log(1-pClass1);
	if (p1>p0):
		return 1;
	return 0;
# lists,classes = loadDataSet();			
# voa = createVocList(lists);
# trainSet = [];
# for data in lists:
	# trainSet.append(setOfWord2Vec(voa,data));
# a,b,c = trainNB0(trainSet,classes);
# v1 = ['love','my','aaaa'];
# v2 = ['stupid','garbage'];
# print('test',v1);
# t = setOfWord2Vec(voa,v1);
# print(classifyNB(t,a,b,c));
# print('test',v2);
# t = setOfWord2Vec(voa,v2);
# print(classifyNB(t,a,b,c));

def setOfWord2VecMN(vocList,inputSet):
	returnVec=[0]*len(vocList);
	for word in inputSet:
		if word in vocList:
			returnVec[vocList.index(word)] = +1;
		else:
			print("the word %s is not in my vocSet" %word);
	return returnVec;

def textParse(bigString):
	import re;
	listofTokens = re.split(r'\W*',bigString);
	return [tok.lower() for tok in listofTokens if len(tok) > 2];

def spanTest():
	docList=[];
	classList=[];
	fullText=[];
	for i in range (1,26):
		wordList = textParse(open('./spam/%d.txt' %i).read());
		docList.append(wordList);
		fullText.extend(wordList);
		classList.append(1);
		wordList = textParse(open('./ham/%d.txt' %i).read());
		docList.append(wordList);
		fullText.extend(wordList);
		classList.append(0);
	voaList = createVocList(docList);
	trainingSet = [i for i in range(50)];
	testSet=[];
	for i in range(10):
		randIndex = int(np.random.uniform(0,len(trainingSet)));
		testSet.append(trainingSet[randIndex]);
		del(trainingSet[randIndex]);
	trainMat=[];
	trainClasses=[];
	for docIndex in trainingSet:
		trainMat.append(setOfWord2VecMN(voaList,docList[docIndex]));
		trainClasses.append(classList[docIndex]);
	p0v,p1v,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses));
	error = 0;
	for docIndex in  testSet :
		voc = setOfWord2VecMN(voaList,docList[docIndex]);
		tlabel = classifyNB(voc,p0v,p1v,pSpam);
		if(tlabel != classList[docIndex]):
			error+=1;
	print("error rate = ",float(error)/len(testSet));
	
 spanTest();

	
		
		