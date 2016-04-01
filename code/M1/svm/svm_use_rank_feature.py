#coding=utf-8

"""
Author: wepon (http://2hwp.com/)
Code: https://github.com/wepe/DataCastle-Solution

"""


import sys,random,os,cPickle
from multiprocessing import Pool
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.externals import joblib
import pandas as pd

os.mkdir('model')
os.mkdir('preds')


#load data
test = pd.read_csv("../../data/test_x_rank.csv")
test_uid = test.uid
test = test.drop("uid",axis=1)
test_x = test/5000.0
        
train_x = pd.read_csv("../../data/train_x_rank.csv")
train_y = pd.read_csv("../../data/train_y.csv")
train_xy = pd.merge(train_x,train_y,on='uid')
y = train_xy.y
train_xy = train_xy.drop(["uid",'y'],axis=1)
X = train_xy/15000.0


def pipeline(iteration,C,gamma,random_seed):
    clf = SVC(C=C,kernel='rbf',gamma=gamma,probability=True,cache_size=7000,class_weight='balanced',verbose=True,random_state=random_seed)
    clf.fit(X,y)
    joblib.dump(clf, './model/svm{0}.pkl'.format(iteration))

    pred = clf.predict_proba(test_x)
    test_result = pd.DataFrame(columns=["uid","score"])
    test_result.uid = test_uid
    test_result.score = pred[:,1]
    test_result.to_csv('./preds/svm_pred{0}.csv'.format(iteration),index=None)


if __name__ == "__main__":
    random_seed = range(2016,2046)
    C = [i/10.0 for i in range(10,40)]
    gamma = [i/1000.0 for i in range(1,31)]
    random.shuffle(random_seed)
    random.shuffle(C)
    random.shuffle(gamma)

    with open('./params.pkl','w') as f:
        cPickle.dump((random_seed,C,gamma),f)

    #to reproduce my result, uncomment following lines
    """
    with open('params_for_reproducing.pkl','r') as f:
        random_seed,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight = cPickle.load(f)    
    """

    for i in range(30):
        pipeline(i,C[i],gamma[i],random_seed[i])
        
        
    #mp version
    """
    pool = Pool(4)
    for i in range(30):
        pool.apply_async(pipeline,args=(i,C[i],gamma[i],random_seed[i]))
    pool.close()
    pool.join()
    """