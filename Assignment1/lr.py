# -*- coding: utf-8 -*-
"""
Created on Tue Dec 09 21:54:00 2014

@author: wepon

程序说明：
loadData函数
实现的功能是从文件夹中读取所有文件，并将其转化为矩阵返回
如调用loadData('train')，则函数会读取所有的txt文件（'0_0.txt'一直到'1_150.txt'）
并将每个txt文件里的32*32个数字转化为1*1024的矩阵，最终返回大小是m*1024的矩阵
同时返回每个txt文件对应的数字，0或1

sigmoid函数
实现sigmoid函数的功能

gradAscent函数
用梯度下降法计算得到回归系数

classfy函数
根据回归系数对输入的样本进行预测

"""
#!/usr/bin/python
from numpy import *
import  numpy as np
from os import listdir
from sklearn.datasets import load_svmlight_file
from sklearn import metrics


def loadDotData(filename):
    data, label = load_svmlight_file(filename)
    data=data.toarray()
    newdata=zeros([data.shape[0],124])
    newdata[0:data.shape[0],0:data.shape[1]]=data
    label[label==-1]=0
    label=label.reshape(-1,1)
    return newdata,label
def metric(y,score):
    print y
    print score
    m=len(score)
    acc=0.0
    for i in range(m):
        if int(score[i])>0.5:
            if int(y[i])==1:
                acc+=1
        else:
            if int(y[i])==0:
                acc+=1

    acc = acc/m

    Y_test=y
    y_score=score
    Y_test2 = np.array(Y_test[:,0].astype(int))
    y_score2 = np.array(y_score[:,0])
    #print(y.shape)
    #print(score.shape)
    
    test_auc = metrics.roc_auc_score(Y_test2,y_score2)
    test_aps = metrics.average_precision_score(Y_test2,y_score2)
    y_score2[y_score2>0.5]=1
    y_score2[y_score2<=0.5]=0
    test_acc = metrics.accuracy_score(Y_test2,y_score2)
    y_pred = np.array(y_score2.astype(int))
    test_f1 = metrics.f1_score(Y_test2,y_pred)
    test_pre = metrics.precision_score(Y_test2,y_pred)
    test_rec = metrics.recall_score(Y_test2,y_pred)
    print(" acc: %.4f - testACC: %.4f  - testAUC: %.4f - testAP: %.4f"\
           " - testF1: %.4f - testPrecision: %.4f - testRecall: %.4f" \
            %( acc, test_acc ,test_auc,test_aps,test_f1,test_pre,test_rec))
    #Vis.plotROC(Y_test2,y_score2,modefilepath)
    return acc ,test_acc ,test_auc, test_aps,test_f1,test_pre,test_rec


def sigmoid(inX):
    #print inX
    return 1.0/(1+np.exp(-inX))

#alpha:步长，maxCycles:迭代次数，可以调整
def gradAscent(dataArray,labelArray,alpha,maxCycles):
    dataMat=mat(dataArray)    #size:m*n
    labelMat=mat(labelArray)      #size:m*1
    m,n=shape(dataMat)
    weigh=ones((n,1)) 
    for i in range(maxCycles):
        h=sigmoid(dataMat*weigh)
        error=labelMat-h    #size:m*1
        weigh=weigh+alpha*dataMat.transpose()*error
    return weigh

def classify(testdir,weigh):
    #dataArray,labelArray=loadData(testdir)
    data,label=loadDotData("a9a/a9a.t")

    dataMat=mat(data)
    labelMat=mat(label)

    h=sigmoid(dataMat*weigh)  #size:m*1
    metric(labelMat,h)

                #print 'error'
    #print 'error rate is:','%.4f' %(1-error/m)
    
    #print "acc:"+str(acc)
    #return acc
                
def digitRecognition(trainDir,testDir,alpha=0.07,maxCycles=10):
    #data,label=loadData(trainDir)
    data,label=loadDotData("a9a/a9a")
    

    
    #print label
    #data = np.asarray(data)
    #data=data.astype(float)
    weigh=gradAscent(data,label,alpha,maxCycles)
    return classify(testDir,weigh)


y_pred = [0, 0.6, 0.1, 0.8]
y_true = [0, 1, 0, 1]
metric(np.asarray(y_true).reshape(-1,1),np.asarray(y_pred).reshape(-1,1))
import sys
sys.exit()
trainDir="train"
testDir="test"
for i in range(0,18):
    #a=2*np.power(0.1,i)
    a=0.0018+ (i*0.00001)
    digitRecognition(trainDir,testDir,alpha=a,maxCycles=10) 
    #print str(a)+"\t"+str(acc)   
        
    
    
    
        
    
        
        
        
        
        