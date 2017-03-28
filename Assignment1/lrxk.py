#import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.datasets import load_svmlight_file
from sklearn import metrics
def sigmoid(X):
    OVERFLOW_THRESH = -709
    X = np.sum(X)
    return 0.0 if X < OVERFLOW_THRESH else (1.0 / (1.0 + np.exp(-1.0 * X)))
def gradient_descent(_x_arr, _y_arr, _thetas, _alpha=12):
    """ 
    
    :param _x_arr: 
    :param _y_arr: 
    :param _thetas:  
    :param _alpha: 
    :return: 
    """
    _thetas_new = _thetas.copy()
    for j in range(_x_arr.shape[1]):
        sum_err = 0.0
        for i in range(_x_arr.shape[0]):
            sum_err += (sigmoid((np.matrix(_thetas) * np.matrix(_x_arr[i]).T).item(0, 0)) - _y_arr[i]) * _x_arr[i,j]
        _thetas_new[j] = _thetas[j] - _alpha * sum_err / _x_arr.shape[0]
    return _thetas_new,sum_err

def IRLS(_x_arr, _y_arr, _thetas):
    """ 

  
    """
    #print _x_arr.shape
    #print _y_arr.shape
    bias = np.eye(_x_arr.shape[1])*0.000001
    sum_gd = np.zeros([_x_arr.shape[1],1])
    sum_hs = np.matrix(np.zeros((_x_arr.shape[1], _x_arr.shape[1])))
    for i in range(_x_arr.shape[0]):
        _x_i_mat = np.matrix(_x_arr[i]).T
        g = sigmoid((np.matrix(_thetas) * _x_i_mat).item(0, 0))
        sum_gd += (_y_arr[i]-g ) * _x_i_mat # (\mu - y) * x
        #print sum_gd.shape
        sum_hs += -((g * (1 - g)) * _x_i_mat * _x_i_mat.T+bias) # # \mu (1- \mu) * X.T * X
        
    gd = sum_gd / _x_arr.shape[0]  # 
    hs = sum_hs / _x_arr.shape[0]  # hession
    
    _thetas -= np.array(hs.I * gd).ravel()
    
    return _thetas

def IRLSL2(_x_arr, _y_arr, _thetas,w_lambda):
    """ +L2-norm

    """
    
    IL2 = np.eye(_x_arr.shape[1])*w_lambda
    
    bias = np.eye(_x_arr.shape[1])*0.000001
    sum_gd = np.zeros([_x_arr.shape[1],1])
    
    sum_hs = np.matrix(np.zeros((_x_arr.shape[1], _x_arr.shape[1])))
    for i in range(_x_arr.shape[0]):
        _x_i_mat = np.matrix(_x_arr[i]).T
        g = sigmoid((np.matrix(_thetas) * _x_i_mat).item(0, 0))
        #sum_gd += (g- _y_arr[i]) * _x_i_mat - (IL2).dot(_thetas).reshape(-1,1) 
        #print ((_x_i_mat * _x_i_mat.T+IL2).I).shape
        sum_gd += (y_arr[i]-g) * _x_i_mat - IL2*(_x_i_mat * _x_i_mat.T+ bias).I * _x_i_mat * _y_arr[i]
        sum_hs += - (g * (1 - g)) * _x_i_mat * _x_i_mat.T - w_lambda*IL2
        
    gd = sum_gd / _x_arr.shape[0]  # gd
    hs = sum_hs / _x_arr.shape[0]  # hession
    
    _thetas -= np.array(hs.I * gd).ravel()
    
    return _thetas


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

def loadDotData(filename):
    data, label = load_svmlight_file(filename)
    #label=label.toarray()
    data=data.toarray()
    newdata=np.zeros([data.shape[0],124])
    newdata[0:data.shape[0],0:data.shape[1]]=data
    label[label==-1]=0
    #label=label.reshape(-1,1)
    return newdata,label
def lossfun(x_arr,y_arr,thetas_nt):
    loss = 0
    for i in range(x_arr.shape[0]):
        wx=(np.matrix(thetas_nt) * np.matrix(x_arr[i]).T).item(0, 0)
        loss += np.log(0.000001+sigmoid(wx)) * y_arr[i] +\
        np.log(1 - sigmoid(wx)) * (1 - y_arr[i])
    loss = -loss / x_arr.shape[0]
    return loss

def lossfun2(x_arr,y_arr,thetas_nt):
    loss = 0
    for i in range(x_arr.shape[0]):
        wx=(np.matrix(thetas_nt) * np.matrix(x_arr[i]).T).item(0, 0)
        loss += wx * y_arr[i] + 1 - sigmoid(wx)
    loss = loss / x_arr.shape[0]
    return loss

def lossfun3(x_arr,y_arr,thetas_nt):
    loss = 0
    wx=(thetas_nt.T * x_arr)
    #print wx.shape
    loss = y_arr* wx.T  + 1 - sigmoid(wx)
    #print loss.shape
    loss = np.mean(loss) 
    return loss

def lossfun4(x_arr,y_arr,thetas_nt):
    loss = 0
    for i in range(x_arr.shape[0]):
        wx=(np.matrix(thetas_nt) * np.matrix(x_arr[i]).T).item(0, 0)
        #print wx.shape
        loss += wx * y_arr[i] + 1 - sigmoid(wx)
    loss = loss / x_arr.shape[0]
    return loss


def lossL2Normfun(x_arr,y_arr,thetas,w_lambda):
    loss = 0
    for i in range(x_arr.shape[0]):
        l2norm =np.linalg.norm(thetas,2)
        wx=(np.matrix(thetas) * np.matrix(x_arr[i]).T).item(0, 0)
        loss += np.log(0.000001+sigmoid(wx)) * y_arr[i] +\
        np.log(1 - sigmoid(wx)) * (1 - y_arr[i])
    loss = -(loss / x_arr.shape[0] -0.5*w_lambda * l2norm)
    return loss

def lossL2Normfun2(x_arr,y_arr,thetas,w_lambda):
    loss = 0
    for i in range(x_arr.shape[0]):
        l2norm =np.linalg.norm(thetas,2)
        wx=(np.matrix(thetas) * np.matrix(x_arr[i]).T).item(0, 0)
        loss += wx * y_arr[i] +1 - sigmoid(wx) 

    loss = loss / x_arr.shape[0] - 0.5* w_lambda * l2norm
    return loss

x_arr,y_arr=loadDotData("a9a/a9a")
x_arr=x_arr[0:100,:]
y_arr=y_arr[0:100]
#x_test,y_test=loadDotData("a9a/a9a.t")
niteration =5
bias = np.eye(x_arr.shape[1])*0.000001
thetas_gd =  np.zeros([x_arr.shape[1]])  # IRLS
thetas_nt =  np.zeros([x_arr.shape[1]])  # IRLS
thetas_nt2 =  np.zeros([x_arr.shape[1]])  # IRLS
thetas_ntL2 =  np.zeros([x_arr.shape[1]])  # IRLS-L2
thetas_ntL22 =  np.zeros([x_arr.shape[1]])  # IRLS-L2
err_gd_list = [0.0] * niteration  # 
err_nt_list = [0.0] * niteration  # 
err_nt_list2 = [0.0] * niteration  # 
err_ntL2_list = [0.0] * niteration  # 
err_ntL2_list2 = [0.0] * niteration  # 

w_lambda= 0.0003
for j in range(niteration):
    #thetas_gd,err_gd_list[j] =gradient_descent(x_arr,y_arr,thetas_gd)
    #print err_gd_list[j]
    
    #IRLS
    err_nt_list[j] = lossfun4(x_arr,y_arr,thetas_nt)
    thetas_nt = IRLS(x_arr, y_arr, thetas_nt.copy())

    #err_nt_list2[j] = lossfun3(x_arr,y_arr,thetas_nt2)
    #thetas_nt2 = IRLS(x_arr, y_arr, thetas_nt2.copy())
    err_ntL2_list[j] = lossL2Normfun2(x_arr,y_arr,thetas_ntL2,w_lambda)
    thetas_ntL2 = IRLSL2(x_arr, y_arr, thetas_ntL2.copy(),w_lambda)
    #err_ntL2_list2[j] = lossL2Normfun2(x_arr,y_arr,thetas_ntL22)
    #thetas_ntL22 = IRLSL2(x_arr, y_arr, thetas_ntL22.copy(),w_lambda)
    #print(j,err_nt_list2[j],err_ntL2_list2[j])
    #print(j,err_nt_list[j],err_ntL2_list[j])
#print err_gd_list
print err_nt_list
print err_ntL2_list




