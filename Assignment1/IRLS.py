#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
# Author: Xu KUI, xukui.cs@gmail.com
# Created Time : Wed 24 Mar 2017 22:05:03 AM CST
# Last Modified: Wed 30 Mar 2017 02:18:00 AM CST
# File Name: IRLS.py
# Description: Iterative Reweighted Logistic Regression with L2 norm

"""

import numpy as np
from sklearn import metrics
from sklearn.datasets import load_svmlight_file

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-1.0 * z))
   
def IRLS(_x_arr, _y_arr, _thetas):
    """ 
    (XRX^T)^{-1}XRz
    R=u(1-u)
    z=X^T W_t + R^{-1}(y-u)
  
    """
    bias = np.eye(_x_arr.shape[1])*0.000001
    loss = np.zeros([_x_arr.shape[1],1])
    
    X=_x_arr.T
    W=_thetas
    y=_y_arr
    WtX = W.T .dot(X)

    u = np.array(sigmoid(np.matrix(WtX)))
    R = u*(1-u)
    z= X.T .dot(W)- (np.power(R,-1)*(u-y)).T
    XRXt=(X*R).dot(X.T)

    left= np.array(np.matrix(XRXt+bias).I)
    loss = (left .dot(X) *R).dot(z)

    _thetas = loss
    return _thetas

def IRLS_L2(_x_arr, _y_arr, _thetas,w_lambda=0.000000001):
    """ 
    (XRX^T + \lambda I)^{-1}XRz
    R=u(1-u)
    z=X^T W_t + R^{-1}(y-u)
  
    """

    lambdaI = np.eye(_x_arr.shape[1])*w_lambda
    
    
    X=_x_arr.T
    W=_thetas
    y=_y_arr
    WtX = W.T .dot(X)

    u = np.array(sigmoid(np.matrix(WtX)))
    R = u*(1-u)
    z= X.T .dot(W)- (np.power(R,-1)*(u-y)).T
    XRXt=(X*R).dot(X.T)
    left= np.array(np.matrix(XRXt+lambdaI).I)
    _thetas = (left .dot(X) *R).dot(z)
    return _thetas


def metric(y,score):

    Y_test=y
    y_score=score
    Y_test2 = np.array(Y_test[:,0].astype(int))
    y_score2 = np.array(y_score[:,0])
    
    test_auc = metrics.roc_auc_score(Y_test2,y_score2)
    test_aps = metrics.average_precision_score(Y_test2,y_score2)
    y_score2[y_score2>0.5]=1
    y_score2[y_score2<=0.5]=0
    test_acc = metrics.accuracy_score(Y_test2,y_score2)
    y_pred = np.array(y_score2.astype(int))
    test_f1 = metrics.f1_score(Y_test2,y_pred)
    test_pre = metrics.precision_score(Y_test2,y_pred)
    test_rec = metrics.recall_score(Y_test2,y_pred)
    #print(" testACC: %.4f  - testAUC: %.4f - testAP: %.4f"\
    #       " - testF1: %.4f - testPrecision: %.4f - testRecall: %.4f" \
    #       %(  test_acc ,test_auc,test_aps,test_f1,test_pre,test_rec))
    #Vis.plotROC(Y_test2,y_score2,modefilepath)
    return test_acc ,test_auc, test_aps,test_f1,test_pre,test_rec

def loadDotData(filename):
    data, label = load_svmlight_file(filename)
    #label=label.toarray()
    data=data.toarray()
    newdata=np.zeros([data.shape[0],124])
    newdata[0:data.shape[0],0:data.shape[1]]=data
    label[label==-1]=0
    #label=label.reshape(-1,1)
    return newdata,label

def lossfun(x_arr,y_arr,W):
    #wx=thetas.T * x_arr
    #loss = np.log(0.000001+sigmoid(wx)) * y_arr + np.log(1 - sigmoid(wx)) * (1 - y_arr)
    loss = 0
    for i in range(x_arr.shape[0]):
        wx=(W.T * x_arr[i]).item(0, 0)
        loss += np.log(0.000001+sigmoid(wx)) * y_arr[i] +\
        np.log(1 - sigmoid(wx)) * (1 - y_arr[i])
    loss = -loss / x_arr.shape[0]

    y_pred=sigmoid(x_arr.dot(W))  
    y=y_arr.reshape(-1,1)
    Y = np.array(y[:,0].astype(int))
    y_score = np.array(y_pred[:,0])
    y_score[y_score>0.5]=1
    y_score[y_score<=0.5]=0
    acc = metrics.accuracy_score(Y,y_score)

    return loss,acc


def classify(W,x_arr,y_arr):

    y=sigmoid(x_arr.dot(W))  
    y_arr=y_arr.reshape(-1,1)

    l2norm =np.linalg.norm(W,2)
      
    test_acc ,test_auc, test_aps,test_f1,test_pre,test_rec = metric(y_arr,y)
    print(" * testACC: %.4f  - testAUC: %.4f - testAP: %.4f"\
        " - testF1: %.4f - testPrecision: %.4f - testRecall: %.4f - L2norm: %.4f" \
            %(  test_acc ,test_auc,test_aps,test_f1,test_pre,test_rec,l2norm))

def plotLossAcc(trainHistory,title):
    import matplotlib.pyplot as plt
    data=trainHistory
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(data['train_loss'],label="Training Loss")
    ax[0].plot(data['test_loss'],label="Testing Loss")
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Iteration')
    minTestLoss = np.min(data['test_loss'])
    stdTestLoss = np.std(data['test_loss'])
    title0 = "Loss by %s \n Min(test loss)=%.4f Std(test loss)=%.4f \n BestModel Iteration:%d"\
     %(title, minTestLoss,stdTestLoss,data['bestIteration'])



    ax[0].set_title(title0)
    ax[0].legend(loc='center right', shadow=True)
    #error
    ax[1].plot(data['train_acc'],label="Training Accuracy")
    ax[1].plot(data['test_acc'],label="Testing Accuracy")
    ax[1].set_ylabel('Top1 Error')
    ax[1].set_xlabel('Iteration')
    maxTestError = np.max(data['test_acc'])
    stdTestError = np.std(data['test_acc'])
    title1 = "Accuracy by %s \n Max(test Acc)=%.4f Std(test Acc)=%.4f \n BestModel Iteration:%d"\
     %(title, maxTestError,stdTestError,data['bestIteration'])

    ax[1].set_title(title1)

    ax[1].legend(loc='center right', shadow=True)
    fig.tight_layout()

    plt.show() 

def train(x_arr,y_arr,x_test,y_test,maxIteration=50,stopEarly=10,regularizer="",w_lambda=0.000000001):

    print "===== IRLS "+regularizer+" ===="
    print "Max Iteration:\t%s" % (maxIteration)
    print "Early Stopping:\t%s" % (stopEarly)
    if regularizer=="L2" : print("lambda:\t%s" % (w_lambda)) 
    print ""
    thetas_nt =  np.zeros([x_arr.shape[1],1])  # IRLS

    trainHistory = {"train_loss":[],"train_acc":[],"test_loss":[],"test_acc":[],"bestIteration":0}
    noImproveCount=0
    minLoss=99999
    bestModel=99999
    for j in range(maxIteration):
        
        #IRLS
        train_loss,train_acc = lossfun(x_arr,y_arr,thetas_nt)
        trainHistory["train_loss"].append(train_loss)
        trainHistory["train_acc"].append(train_acc)
        #
        thetas_nt = IRLS_L2(x_arr, y_arr, thetas_nt.copy(),w_lambda)


        test_loss,test_acc = lossfun(x_test,y_test,thetas_nt)
        trainHistory["test_loss"].append(test_loss)
        trainHistory["test_acc"].append(test_acc)
        info="model is not improved"
        if minLoss > test_loss :
            info="best model from {:f} to {:f} ".format(minLoss,test_loss)        
            minLoss = test_loss
            bestModel = thetas_nt        
            np.save("model/IRLS"+regularizer+"_bestmodel.npy",bestModel)
        else:
            noImproveCount +=1
        print ("[ %d/%d] %s" %(j,maxIteration,info))
        print (" | trainLoss %.4f   trainAcc %.4f   valLoss %.4f   valAcc %.4f"%(train_loss,train_acc,test_loss,test_acc ))
        if noImproveCount >stopEarly:
            print "Stoped by earlystoping, best model loss: %.4f in Iteration %d " %(minLoss,j+1-stopEarly)
            break
    print ""
    trainHistory["bestIteration"] = j+1-stopEarly
    return trainHistory

def test(x_test,y_test,regularizer=""):
    # test
    print " Testing IRLS "+regularizer+" Loading Best model..."
    model=np.load("model/IRLS"+regularizer+"_bestmodel.npy")
    classify(model,x_test,y_test)

if __name__ == '__main__':
    
    x_train, y_train=loadDotData("a9a/a9a")
    x_test,y_test=loadDotData("a9a/a9a.t")
    maxIteration =50

    # IRLS 
    trainHistory=train(x_train, y_train,x_test,y_test)
    test(x_test,y_test,regularizer="")
    plotLossAcc(trainHistory,"IRLS on a9a data")

    # IRLS L2 norm
    trainHistoryL2=train(x_train, y_train,x_test,y_test,regularizer="L2",w_lambda= 0.3)
    test(x_test,y_test,regularizer="L2")


