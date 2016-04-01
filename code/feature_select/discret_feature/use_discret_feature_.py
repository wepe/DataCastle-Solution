#coding=utf-8

"""
Author: wepon (http://2hwp.com/)
Code: https://github.com/wepe/DataCastle-Solution

"""

from sklearn.cross_validation import train_test_split
import pandas as pd
import xgboost as xgb
import sys,random
import cPickle
import os

os.mkdir('featurescore')
os.mkdir('model')
os.mkdir('preds')


#use discret feature 'dx*' and 'n1'~'n10'
#load data
train_x_d = pd.read_csv('../../data/train_x_discretization.csv')
train_x_nd = pd.read_csv('../../data/train_x_nd.csv')
train_x = pd.merge(train_x_d,train_x_nd,on='uid')
train_y = pd.read_csv('../../data/train_y.csv')
train_xy = pd.merge(train_x,train_y,on='uid')
y = train_xy.y
X = train_xy.drop(['uid','y'],axis=1)
dtrain = xgb.DMatrix(X, label=y)

    
test_x_d = pd.read_csv('../../data/test_x_discretization.csv')
test_x_nd = pd.read_csv('../../data/test_x_nd.csv')
test = pd.merge(test_x_d,test_x_nd,on='uid')
test_uid = test.uid
test_x = test.drop("uid",axis=1)
dtest = xgb.DMatrix(test_x)


def pipeline(iteration,random_seed,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight):
    if max_depth==6:
        num_boost_round = 550
    elif max_depth==7:
        num_boost_round = 450
    elif max_depth==8:
        num_boost_round = 400
    
    params={
    	'booster':'gbtree',
    	'objective': 'rank:pairwise',
    	'scale_pos_weight': float(len(y)-sum(y))/float(sum(y)),
        'eval_metric': 'auc',
    	'gamma':gamma,
    	'max_depth':max_depth,
    	'lambda':lambd,
        'subsample':subsample,
        'colsample_bytree':colsample_bytree,
        'min_child_weight':min_child_weight, 
        'eta': 0.04,
    	'seed':random_seed
        }
    
    watchlist  = [(dtrain,'train')]
    model = xgb.train(params,dtrain,num_boost_round=num_boost_round,evals=watchlist)
    model.save_model('./model/xgb{0}.model'.format(iteration))
    
    #predict test set
    test_y = model.predict(dtest)
    test_result = pd.DataFrame(test_uid,columns=["uid"])
    test_result['score'] = test_y
    test_result.to_csv("./xgbs/xgb{0}.csv".format(iteration),index=None,encoding='utf-8')
    
    #get feature score
    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
    fs = []
    for (key,value) in feature_score:
        fs.append("{0},{1}\n".format(key,value))
    
    with open('./featurescore/feature_score_{0}.csv'.format(iteration),'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)



if __name__ == "__main__":
    random_seed = range(1000,2000,20)
    gamma = [i/1000.0 for i in range(100,200,2)]
    max_depth = [6,7,8]
    lambd = range(200,400,2)
    subsample = [i/1000.0 for i in range(600,700,2)]
    colsample_bytree = [i/1000.0 for i in range(250,350,2)]
    min_child_weight = [i/1000.0 for i in range(200,300,2)]
    random.shuffle(random_seed)
    random.shuffle(gamma)
    random.shuffle(max_depth)
    random.shuffle(lambd)
    random.shuffle(subsample)
    random.shuffle(colsample_bytree)
    random.shuffle(min_child_weight)
    
    with open('params.pkl','w') as f:
        cPickle.dump((random_seed,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight),f)

    #to reproduce my result, uncomment following lines
    """
    with open('params_for_reproducing.pkl','r') as f:
        random_seed,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight = cPickle.load(f)    
    """
    
    for i in range(36):
        pipeline(i,random_seed[i],gamma[i],max_depth[i%3],lambd[i],subsample[i],colsample_bytree[i],min_child_weight[i])
