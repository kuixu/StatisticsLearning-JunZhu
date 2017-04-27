import numpy as np
from scipy.misc import logsumexp

def loadData(libfilepath, vocfilepath):
    print "Loading data......"
    X = []
    #Loading word file
    wordDict = {}
    vocfile = open(vocfilepath,"r")
    for line in vocfile.readlines():
        words = line.split('\t')
        wordDict[int(words[0])] = words[1]
    vocfile.close()
    # Loading lib file
    wordSize = len(wordDict)
    libfile = open(libfilepath,"r")
    for line in libfile.readlines():
        doc = np.zeros(wordSize, dtype='float')
        words = line.strip().split('\t')[1].split(" ")
        for word in words:
            word = word.split(":")
            doc[int(word[0])] = int(word[1])
        X.append(doc)
    libfile.close()
    return np.asarray(X), wordDict

def MultinomialMixtureEM(X, K, P_Cd=None, eps=0.00001, maxiter=1000):
    '''Fit a multinomial mixture model using EM.
       Parameters
       ----------
       X : 2D array of float
           Training data, an instance on each row.
       K : int
           Number of mixture components.
       pz : 2D array of float
           Initial component probabilities P(z|x), one instance x on each row,
           one component z on each column. If None, generate the initial
           probabilities randomly.
       eps : float
           Stop when change in P(z|x) is less than eps.
       maxiter : int
           Maximum number of iterations.
       Returns
       -------
       phi : 2D array of float
           Component distributions, one component on each row.
       P_Cd : 2D array of float
           P(z|x), an instance x on each row, a component z on each column.
    '''
    D = X.shape[0]
    Nd = np.sum(X, axis=1)
    if P_Cd is None:
        #P_Cd = np.random.dirichlet(np.ones(K), D)
      P_Cd = np.random.rand( D, K)

    convergince = np.finfo('float').max
    i = 0
    while i < maxiter and convergince > eps:
        theta = np.sum(P_Cd, axis=0) / D
        phi = (X.T.dot(P_Cd) / np.sum(P_Cd.T * Nd, axis=1)).T
        L = np.log(theta) + X.dot(np.log(phi.T))
        
        P_Cd_new = np.exp((L.T - logsumexp(L, axis=1)).T)
        convergince = np.max(np.abs(P_Cd_new-P_Cd))
        P_Cd = P_Cd_new
        i += 1

    return phi, P_Cd

def printResults(phi, P_Cd, K,NumMostFrequent, X, wordDict):
    (D,W)=X.shape
    #NumMostFrequent =5
    word = [[] for i in range(K)]
    #f = file('./nips/output.txt', 'w+')
    print "The K is: %d" %(K)
    for i in range(K):
        #print "The cluster %d\'s ratio is %f, most-frequent %d words are: " % (i+1, self.PI[i], num)
        #print>>f, "The cluster %d\'s ratio is %f, most-frequent %d words are: " % (i+1, self.PI[i], num)
        list = []
        for j in range(W):
            list.append((phi[i,j],j))
        list.sort(key=lambda x:x[0], reverse=True)
        for j in range(NumMostFrequent):
            word[i].append(wordDict[list[j][1]])
            print wordDict[list[j][1]],
            #print>>f, dataset.voc_dict[list[j][1]]
        print ""


    sim=0.0
    count = 0.0
    for i in range(K-1):
        for j in range(i+1,K):
            cnt = 0.0
            for k in range(5):
                for l in range(5):
                    if word[i][k] is word[j][l]:
                        cnt = cnt+1.0
            # print cnt
            count = count+1.0
            sim = sim+cnt / 5.0
    sim = sim/count
    return sim



Kset = [5,10,20,30]
index = 0
NumMostFrequent = 5
bestKsim= 1
for i in range(len(Kset)):
    K = Kset[i]
    
    
    X, wordDict = loadData("./nips/nips.libsvm", "./nips/nips.vocab")
    #Dataset = Input.Dataset(voc_path=voc_file, lib_path=lib_file)
    #print "Dataset done......"
    #X= np.asarray(Dataset.x_)
    (D,W)=X.shape
    phi, P_Cd = MultinomialMixtureEM(X,K)
    print phi.shape
    print P_Cd.shape
    sim = printResults(phi, P_Cd, K, NumMostFrequent, X, wordDict)
    if bestKsim > sim:
        bestKsim= sim
        index = i
    print "Average similarity is %f " %(sim)
    #print "Building model......"
    #MoM = model.model(cluster_num, Dataset.num_lib, Dataset.voc_size)
    #print "Model done......"

    #MoM.train(x_=np.array(Dataset.x_, dtype='float'), iter_num=20)

    #MoM.output(5, dataset=Dataset)

    #if MoM.sim < mn:
        #mn = MoM.sim
        #index = i
print ""
print "The best K is: %d " %(Kset[index])



# voc_file = "./nips/nips.vocab"
# lib_file = "./nips/nips.libsvm"
#
# # file = open(lib_file)
# print "Geting dataset......"
# Dataset = Input.Dataset(voc_path=voc_file, lib_path=lib_file)
# print "Dataset done......"
#
# print "Building model......"
# MoM = model.model(cluster_num, Dataset.num_lib, Dataset.voc_size)
# print "Model done......"
#
# MoM.train(x_=np.array(Dataset.x_, dtype='float'), iter_num=20)
# MoM.output(5, dataset=Dataset)
