from math import log;
import numpy as np;
import matplotlib.pyplot as plt;
import os;
def clacEnt(dataSet):
	num = len(dataSet);
	labelCount = {};
	for i in range(num):
		label = dataSet[i][-1];
		labelCount[label] =labelCount.get(label,0)+ 1;
	ent = 0.0;
	for item in labelCount:
		p = labelCount[item]/float(num);
		ent += -p*log(p,2);
	return ent;

def createDataSet():
	dataSet=  [[1,1,'yes'],
				[1,1,'yes'],
				[1,0,'no'],
				[0,1,'no'],
				[0,1,'no'],
				[1,1,'mabey']];
	labels= ['no surfacing','flippers'];
	return dataSet,labels;

# dataSet,labels = createDataSet();
# print(clacEnt(dataSet));
def spiltDataSet(dataSet,axis,value):
	retSet=[];
	for item in dataSet:
		if item[axis] == value:
			vec = item[:axis];
			vec.extend(item[axis+1:]);
			retSet.append(vec);
	return retSet;

def chooseBestFeatureToSpilt(dataSet):
	featureNum = len(dataSet[0])-1;
	baseEntropy = clacEnt(dataSet);
	bestInfoGain = 0.0;
	bestFeature = -1;
	for i in range(featureNum):
		itemList = [e[i] for e in dataSet];
		itemSet = set(itemList);
		newEntropy = 0.0
		for item in itemSet:
			spiltSet = spiltDataSet(dataSet,i,item);
			p = len(spiltSet)/float(len(dataSet));
			newEntropy += p*clacEnt(spiltSet);
		infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
		if (infoGain > bestInfoGain):       #compare this to the best gain so far
			bestInfoGain = infoGain         #if better than current best, set to best
			bestFeature = i
	print(bestFeature);
	return bestFeature;

def majorityCnt(classList):
	classCount={};
	for item in classList:
		classCount[item]= classCount.get(item,0)+1;
	result = sorted(classCount.iteritems(),key=lambda dd:dd[1],reverse=True);
	return result[0][0];
		
		
def createTree(dataSet,labels):
	classList = [e[-1] for e in dataSet];
	if classList.count(classList[0]) == len(classList):
		return classList[0];
	if len(dataSet[0]) == 1:
		return majorityCnt(classList[0]);
	bestFeat =  chooseBestFeatureToSpilt(dataSet);
	t=labels[bestFeat];
	myTree ={t:{}};
	del(labels[bestFeat]);
	itemList = [e[bestFeat] for e in dataSet];
	itemSet = set(itemList);
	for item in itemSet:
		subLabels = labels[:];
		myTree[t][item] = createTree(spiltDataSet(dataSet,bestFeat,item),subLabels);
	return myTree;

# dataSet,labels = createDataSet();
# myTree = createTree(dataSet,labels);
# print(myTree);

# decNode = dict(boxstyle="sawtooth",fc="0.8");
# leafNode = dict(boxstyle="round4",fc="0.8");
# arrow_args = dict(arrowstyle="<-");
# def plotNode(nodeTxt,centerPt,parentPt,nodeType):
	# createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',
				# xytext=centerPt,textcoords='axes fraction',
				# va="center",ha="center",bbox=nodeType,arrowprops=arrow_args);
	

# def createPlot():
	# fig=plt.figure(1,facecolor='white');
	# fig.clf();
	# createPlot.ax1=plt.subplot(111,frameon=False);
	# plotNode('a decision node',(0.5,0.1),(0.1,0.5),decNode);
	# plotNode('a leaf node',(0.8,0.1),(0.3,0.8),leafNode);
	# plt.show();

# createPlot();	

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
    
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  #this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]     #the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes   
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
# if you do get a dictonary you know it's a tree, and the first element will be another dict

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()

# def createPlot():
   # fig = plt.figure(1, facecolor='white')
   # fig.clf()
   # createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
   # plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
   # plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
   # plt.show()

def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

def classify(inputTree,featLabels,testVec):
	firstStr =  inputTree.keys()[0];
	secondDict = inputTree[firstStr];
	featIndex = featLabels.index(firstStr);
	for key in secondDict.keys():
		if testVec[featIndex] == key:
			if type(secondDict[key]).__name__ == 'dict':
				classLabel = classify(secondDict[key],featLabels,testVec);
			else:
				classLabel = secondDict[key];
	return classLabel;
	
# dataSEt, labels = createDataSet();
# myTree = retrieveTree(0);
# print(classify(myTree,labels,[1,1]));

def storeTree(inputTree,filename):
	import pickle;
	fw = open(filename,'w');
	pickle.dump(inputTree,fw);
	fw.close();

def grabTree(filename):
	import pickle;
	fr = op(filename);
	return pickle.load(fr);
	
# dataSEt, labels = createDataSet();
# myTree = retrieveTree(0);
# storeTree(myTree,'a.txt');
# print(classify(myTree,labels,[1,1]));

def test():
	fr = open('lenses.txt');
	lenses=[inst.strip().split('\t') for inst in fr.readlines()];
	lensesLabels=['age','prescipt','astigmatic','tearRate'];
	lensesTree = createTree(lenses,lensesLabels);
	print(lensesTree);
	createPlot(lensesTree);

test();
				
		
		