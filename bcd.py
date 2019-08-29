
import warnings
warnings.filterwarnings('ignore')
MC1 = 30
MC2 = 50

# Cloning Git Repo
! git clone https://github.com/adityasamant/Breast-Cancer-Detection


import pandas as pd
import numpy as np
data = pd.read_csv("wdbc.data",header=None)
data = data.drop(data.columns[0],axis = 1)

data

X = data.drop(data.columns[0],axis=1)
y = data[1]

X

y


from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,pairwise_distances
def calcmetric(y,yp,ys):
    acc = accuracy_score(y,yp)
    print("Accuracy:",acc)
    prec = precision_score(y,yp,pos_label="M")
    print("Precision:",prec)
    rec = recall_score(y,yp,pos_label="M")
    print("Recall:",rec)
    fsc = f1_score(y,yp,pos_label="M")
    print("F1 Score:",fsc)
    auc = roc_auc_score(y,ys)
    print("Auc:",auc)
    return acc,prec,rec,fsc,auc

def evaluation(y1,yp1,ys1,y2,yp2,ys2):
    
    # Train Data
    print("Train Data")
    acctr,prectr,rectr,fsctr,auctr = calcmetric(y1,yp1,ys1)
    
    # Test Data
    print("Test Data")
    accte,precte,recte,fscte,aucte = calcmetric(y2,yp2,ys2)
    
    return [acctr,prectr,rectr,fsctr,auctr,accte,precte,recte,fscte,aucte]

def farline(dlist,data,y):
    i = np.argmax(dlist)
    return data.iloc[[i]],y.iloc[[i]],data.drop(data.index[i]),y.drop(y.index[i])

def decfunc(data,cluster):
    dist = pairwise_distances(data, n_jobs=-1, metric="euclidean")
    for i in range(dist.shape[0]):
        dist[i, i] = np.inf
    dist_0 = np.amin(dist[:, np.argwhere(cluster == 0).ravel()], axis=1).reshape(-1, 1)
    dist_1 = np.amin(dist[:, np.argwhere(cluster == 1).ravel()], axis=1).reshape(-1, 1)
    return np.concatenate((dist_0, dist_1), axis=1)

def plotit(true,pred,prob,truete,predte,probte):
    print("Confusion Matrix for Train Data")
    print(confusion_matrix(true, pred))
    # plot ROC Curve
    fpr, tpr, threshold = roc_curve(true, prob,pos_label="M")
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic Train Data')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.03, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    print("Confusion Matrix for Test Data")
    print(confusion_matrix(truete, predte))
    # plot ROC Curve
    fpr, tpr, threshold = roc_curve(truete, probte,pos_label="M")
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic Test Data')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.03, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering


comparei = []
iFlag = True
    
print("\n\033[1mSupervised Learning\033[0m")
for mc in range(MC1):
    
    print("\n\n\033[91mIteration",mc+1,"\033[0m")
    NX = pd.DataFrame(normalize(X),index=X.index)
    trainX,testX,trainy,testy = train_test_split(NX,y,train_size=0.8,random_state=np.random.randint(1000),stratify=y)
    train = pd.concat([trainX,trainy],axis=1)
    test = pd.concat([testX,testy],axis=1)
    
    print("Shapes of normalized/raw/Split Data")
    print(trainy.shape)
    print(testy.shape)    
    print(trainX.shape)
    print(testX.shape)
    print(train.shape)
    print(test.shape)
    
    bibestc = []
    c = 0.00001
    while c < 10003:
        bi_model = LinearSVC(penalty='l1',C=c,dual=False)
        score = cross_val_score(bi_model,trainX,trainy,cv=5,n_jobs=-1)
        bibestc.append(score.mean())
        c = c * 10
        
    bibestC = 0.00001 * 10**bibestc.index(max(bibestc))
    print("Best C is",bibestC,"with CV score",max(bibestc))
    
    bi_model = LinearSVC(penalty='l1',C=bibestC,dual=False)
    bipredtr = bi_model.fit(trainX,trainy).predict(trainX)
    bipredte = bi_model.predict(testX)
    bidftr = bi_model.decision_function(trainX)
    bidfte = bi_model.decision_function(testX)
    
    comparei.append(evaluation(trainy,bipredtr,bidftr,testy,bipredte,bidfte))
    
    if(iFlag):
        plotit(trainy,bipredtr,bidftr,testy,bipredte,bidfte)
        iFlag = False

bi = np.mean(comparei,axis = 0)
bi


compareii = []
iiFlag = True
print("\n\033[1mSemi Supervised Learning\033[0m")
for mc in range(MC1):
    
    print("\n\n\033[91mIteration",mc+1,"\033[0m")
    NX = normalize(X)
    trainX,testX,trainy,testy = train_test_split(pd.DataFrame(NX),y,train_size=0.8,random_state=np.random.randint(1000),stratify=y)
    
    print("Shapes of normalized/raw/Split Data")
    print(trainy.shape)
    print(testy.shape)    
    print(trainX.shape)
    print(testX.shape)
    print(train.shape)
    print(test.shape)
    
    labelX,unlabelX,labely,unlabely = train_test_split(trainX,trainy,train_size=0.5,random_state=np.random.randint(1000),stratify=trainy)

    print("Shapes of normalized/raw/Split Data")
    print(labelX.shape)
    print(labely.shape)    
    print(unlabelX.shape)
    print(unlabely.shape)
    
    biibestc = []
    c = 0.00001
    while c < 10003:
        bii_model = LinearSVC(penalty='l1',C=c,dual=False)
        score = cross_val_score(bii_model,labelX,labely,cv=5,n_jobs=-1)
        biibestc.append(score.mean())
        c = c * 10
        
    biibestC = 0.00001 * 10**biibestc.index(max(biibestc))
    print("Best C is",biibestC,"with CV score",max(biibestc))
    
    bii_model = LinearSVC(penalty='l1',C=biibestC,dual=False)
    bii_model.fit(labelX,labely)

    while(not unlabelX.size == 0):
        newdata,newdatay, unlabelX, unlabely = farline(np.divide(bii_model.decision_function(unlabelX),np.linalg.norm(bii_model.coef_)),unlabelX,unlabely)
        labelX.append(newdata, ignore_index = True)
        labely.append(pd.DataFrame(bii_model.predict(newdata),index = newdatay.index), ignore_index = True)
        bii_model.fit(labelX,labely)
    
    biipredtr = bii_model.predict(labelX)
    biipredte = bii_model.predict(testX)
    biidftr = bii_model.decision_function(labelX)
    biidfte = bii_model.decision_function(testX)
    
    compareii.append(evaluation(labely,biipredtr,biidftr,testy,biipredte,biidfte))
    
    if(iiFlag):
        plotit(labely,biipredtr,biidftr,testy,biipredte,biidfte)
        iiFlag = False
        
bii = np.mean(compareii,axis = 0)
bii

iiiFlag = True
compareiii = []
print("\n\033[1mUnsupervised Learning\033[0m")
for mc in range(MC1):
    
    print("\n\n\033[91mIteration",mc+1,"\033[0m")
    NX = pd.DataFrame(normalize(X),index=X.index)
    trainX,testX,trainy,testy = train_test_split(NX,y,train_size=0.8,random_state=np.random.randint(1000),stratify=y)
    
    km = KMeans(n_clusters=2, random_state=np.random.randint(1000))
    kmean = km.fit_predict(trainX)
    temp = pd.DataFrame(trainy.copy())
    temp['cluster'] = kmean
    
    print("Cluster centers are",km.cluster_centers_)

    # B
    tempdf = km.transform(trainX)
    
    dist = []
    j=0
    for i in kmean:
        dist.append(tempdf[j][i])
        j=j+1
        
    temp['dist']=dist
    
    c = []
    for cluster in range(2):
        classlist = temp.loc[temp['cluster']==cluster].nsmallest(30,'dist')[1].tolist()
        c.append(max(classlist,key=classlist.count))    
    print("Clusters:",c)
    
    biiipredtr = []
    biiidftr = []
    j = 0
    for i in kmean:
        biiidftr.append(1-(tempdf[j][1-i]/(tempdf[j][0]+tempdf[j][1])))
        biiipredtr.append(c[i])
        j = j + 1
    
    # C
    kmeante = km.predict(testX)
    tempte = testy.copy()
    tempte['cluster'] = kmeante
    
    biiipredte = []
    biiidfte = []
    tempdf = km.transform(testX)
    j = 0
    for i in kmeante:
        biiidfte.append(1-(tempdf[j][1-i]/(tempdf[j][0]+tempdf[j][1])))
        biiipredte.append(c[i])
        j = j + 1
        
    
    compareiii.append(evaluation(trainy,biiipredtr,biiidftr,testy,biiipredte,biiidfte))
    
    if(iiiFlag):
        plotit(trainy,biiipredtr,biiidftr,testy,biiipredte,biiidfte)
        iiiFlag = False
        
biii = np.mean(compareiii,axis = 0)
biii


compareiv = []
ivFlag = True
print("\n\033[1mSpectral Clustering\033[0m")

for mc in range(MC1):
    
    print("\n\n\033[91mIteration",mc+1,"\033[0m")
    
    trainX,testX,trainy,testy = train_test_split(X,y,train_size=0.8,random_state=np.random.randint(1000),stratify=y)
    NtrainX = normalize(trainX)
    NtestX = normalize(testX)
    train = pd.concat([trainX,trainy],axis=1)
    test = pd.concat([testX,testy],axis=1)
    Ntrain = pd.concat([pd.DataFrame(NtrainX,index=trainX.index),trainy],axis=1)
    Ntest = pd.concat([pd.DataFrame(NtestX,index=testX.index),testy],axis=1)
    
    sc = SpectralClustering(n_clusters=2, gamma=1.0, affinity="rbf",random_state=np.random.randint(1000)).fit(NtrainX)
    cls = sc.labels_
    clste = sc.fit_predict(NtestX)
    temp = trainy.copy()
    temp['cluster'] = cls
    
    # B
    c = []
    for cluster in range(2):
        classlist = temp.loc[temp['cluster']==cluster].tolist()
        c.append(max(classlist,key=classlist.count))    
    print("Clusters:",c)
    
    bivpredtr = []
    bivdftr = []
    tempdf = decfunc(NtrainX,cls)
    j = 0
    for i in cls:
        bivdftr.append(1-(tempdf[j][1-i]/(tempdf[j][0]+tempdf[j][1])))
        bivpredtr.append(c[i])
        j = j + 1
    
    # C
    tempte = testy.copy()
    tempte['cluster'] = clste
    
    bivpredte = []
    bivdfte = []
    tempdf = decfunc(NtestX,clste)
    j = 0
    for i in clste:
        bivdfte.append(1-(tempdf[j][1-i]/(tempdf[j][0]+tempdf[j][1])))
        bivpredte.append(c[i])
        j = j + 1
        
    
    compareiv.append(evaluation(trainy,bivpredtr,bivdftr,testy,bivpredte,bivdfte))
    
    if(ivFlag):
        plotit(trainy,bivpredtr,bivdftr,testy,bivpredte,bivdfte)
        ivFlag = False
        
biv = np.mean(compareiv,axis = 0)
biv


compare = [bi.tolist(),bii.tolist(),biii.tolist(),biv.tolist()]
Compare = pd.DataFrame(compare,columns=['Train Accuracy','Train Precision','Train Recall','Train f1 Score','Train Auc','Test Accuracy','Test Precision','Test Recall','Test f1 Score','Test Auc'])
Compare['Method']=['Supervised','Semi Supervised','Unsupervised','Spectral Clustering']
Compare


data = pd.read_csv("data_banknote_authentication.txt",header=None)

data

X = data.drop(data.columns[4],axis=1)
y = data[4]

X

y

def pick10(pool,data):
    temp = data.sample(n=10,random_state=np.random.randint(1000))
    if(len(pool) < 500):
        while(temp[4].tolist().count(1) != 4):
            temp = data.sample(n=10,random_state=np.random.randint(1000))
    else:
        while(temp[4].tolist().count(1) != 5):
            temp = data.sample(n=10,random_state=np.random.randint(1000))
    return pool.append(temp),data.drop(temp.index)

from sklearn.model_selection import KFold

errP = []
print("\n\033[1mPassive Learning\033[0m")
for i in range(MC2):
    print("\n\n\033[91mIteration",i+1,"\033[0m")
    
    trainX,testX,trainy,testy = train_test_split(X,y,train_size=900,random_state=np.random.randint(1000),stratify = y)
    NtrainX = normalize(trainX,axis=0)
    NtestX = normalize(testX,axis=0)
    train = pd.concat([trainX,trainy],axis=1)
    test = pd.concat([testX,testy],axis=1)
    Ntrain = pd.concat([pd.DataFrame(NtrainX,index=trainX.index),trainy],axis=1)
    Ntest = pd.concat([pd.DataFrame(NtestX,index=testX.index),testy],axis=1)

    print("Shapes of normalized/raw/Split Data")
    print(trainy.shape)
    print(testy.shape)    
    print(trainX.shape)
    print(testX.shape)
    
    i_err = []
    trset = train.copy()
    pool = pd.DataFrame()
    while(not trset.size == 0):
        pool,trset = pick10(pool,trset)
        print("\nData Size is",len(pool))
        bi2bestc = []
        c = 0.00001
        while c < 100003:
            bi2_model = LinearSVC(penalty='l1',C=c,dual=False)
            score = cross_val_score(bi2_model,pool.drop(pool.columns[4],axis=1),pool[4],cv=KFold(n_splits=10),n_jobs=-1)
            bi2bestc.append(score.mean())
            c = c * 10

        bi2bestC = 0.00001 * 10**bi2bestc.index(max(bi2bestc))
        print("Best C is",bi2bestC,"with CV score",max(bi2bestc))

        bi2_model = LinearSVC(penalty='l1',C=bi2bestC,dual=False).fit(pool.drop(pool.columns[4],axis=1),pool[4])
        score = bi2_model.score(testX,testy)
        i_err.append(1 - score)
        print("Test Error is",1 - score)
    errP.append(i_err)


def near10(K,pool,data):
    i = sorted(range(len(K)), key=lambda x: K[x])[:10]
    temp = data.iloc[i]
    return pool.append(temp),data.drop(temp.index)

errA = []
print("\n\033[1mActive Learning\033[0m")
for i in range(MC2):
    print("\n\n\033[91mIteration",i+1,"\033[0m")
    
    trainX,testX,trainy,testy = train_test_split(X,y,train_size=900,random_state=np.random.randint(1000),stratify = y)
    NtrainX = normalize(trainX,axis=0)
    NtestX = normalize(testX,axis=0)
    train = pd.concat([trainX,trainy],axis=1)
    test = pd.concat([testX,testy],axis=1)
    Ntrain = pd.concat([pd.DataFrame(NtrainX,index=trainX.index),trainy],axis=1)
    Ntest = pd.concat([pd.DataFrame(NtestX,index=testX.index),testy],axis=1)

    print("Shapes of normalized/raw/Split Data")
    print(trainy.shape)
    print(testy.shape)    
    print(trainX.shape)
    print(testX.shape)
    
    i_err = []
    trset = train.copy()
    pool = pd.DataFrame()
    while(not trset.size == 0):
        if(pool.size==0):
            pool,trset = pick10(pool,trset)
        else:
            pool,trset = near10(np.divide(bii2_model.decision_function(trset.drop(trset.columns[4],axis=1)),np.linalg.norm(bii2_model.coef_)),pool,trset)
        print("\nData Size is",len(pool))
        
        bii2bestc = []
        c = 0.00001
        while c < 100003:
            bii2_model = LinearSVC(penalty='l1',C=c,dual=False)
            if(len(pool) < 41):
                score = cross_val_score(bii2_model,pool.drop(pool.columns[4],axis=1),pool[4],cv=2,n_jobs=-1)
            else:
                score = cross_val_score(bii2_model,pool.drop(pool.columns[4],axis=1),pool[4],cv=10,n_jobs=-1)
            bii2bestc.append(score.mean())
            c = c * 10

        bii2bestC = 0.00001 * 10**bii2bestc.index(max(bii2bestc))
        print("Best C is",bii2bestC,"with CV score",max(bii2bestc))

        bii2_model = LinearSVC(penalty='l1',C=bii2bestC,dual=False).fit(pool.drop(pool.columns[4],axis=1),pool[4])
        score = bii2_model.score(testX,testy)
        i_err.append(1 - score)
        print("Test Error is",1 - score)
    errA.append(i_err)


lcP = np.mean(errP,axis = 0)
lcP

lcA = np.mean(errA,axis = 0)
lcA

# Plotting Test and Train Errors against the Value of K neighbours
plt.plot(range(10,905,10),lcP,'r', label="Passive Learning")
plt.plot(range(10,905,10),lcA,'b', label="Active Learning")
plt.xlabel("Training Instances")
plt.ylabel("Average Test Error")
plt.legend()
plt.show()
