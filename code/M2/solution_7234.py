#coding=utf-8

"""
Author: wepon (http://2hwp.com/)
Code: https://github.com/wepe/DataCastle-Solution

"""


import pandas as pd
import xgboost as xgb
import sys,random
import cPickle
import os

os.mkdir('featurescore')
os.mkdir('model')
os.mkdir('preds')




#离散化特征的计数特征
test_nd = pd.read_csv('../data/test_x_nd.csv')[['uid','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10']]
train_nd = pd.read_csv('../data/train_x_nd.csv')[['uid','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10']]
trainunlabeled_nd = pd.read_csv('../data/train_unlabeled_nd.csv')[['uid','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10']]

#缺失值个数的离散化特征
test_dnull = pd.read_csv('../data/test_x_null.csv')[['uid','discret_null']]
train_dnull = pd.read_csv('../data/train_x_null.csv')[['uid','discret_null']]
trainunlabeled_dnull = pd.read_csv('../data/train_unlabeled_null.csv')[['uid','discret_null']]

#n1~n10，discret_null 这11维特征不做特征选择，先放在一起
eleven_feature = ['n1','n2','n3','n4','n5','n6','n7','n8','n9','n10','discret_null']
test_eleven = pd.merge(test_nd,test_dnull,on='uid')
train_eleven = pd.merge(train_nd,train_dnull,on='uid')
trainunlabeled_eleven = pd.merge(trainunlabeled_nd,trainunlabeled_dnull,on='uid')

del test_dnull,train_dnull,trainunlabeled_dnull
del test_nd,train_nd,trainunlabeled_nd


#离散特征
discret_feature_score = pd.read_csv('./discret_feature_score.csv')
fs = list(discret_feature_score.feature[0:500])
discret_train = pd.read_csv("../data/train_x_discretization.csv")[['uid']+fs]
discret_test = pd.read_csv("../data/test_x_discretization.csv")[['uid']+fs]
discret_train_unlabeled = pd.read_csv("../data/train_unlabeled_discretization.csv")[['uid']+fs]

#排序特征
rank_feature_score = pd.read_csv('./rank_feature_score.csv')
fs = list(rank_feature_score.feature[0:500])
rank_train_x = pd.read_csv("../data/train_x_rank.csv")
rank_train = rank_train_x[fs] / float(len(rank_train_x))
rank_train['uid'] = rank_train_x.uid

rank_test_x = pd.read_csv("../data/test_x_rank.csv")
rank_test = rank_test_x[fs] / float(len(rank_test_x))
rank_test['uid'] = rank_test_x.uid

rank_train_unlabeled_x = pd.read_csv("../data/train_unlabeled_rank.csv")
rank_train_unlabeled = rank_train_unlabeled_x[fs] / float(len(rank_train_unlabeled_x))
rank_train_unlabeled['uid'] = rank_train_unlabeled_x.uid

del rank_train_x,rank_test_x,rank_train_unlabeled_x

#原始特征
raw_feature_score = pd.read_csv('./raw_feature_score.csv')
fs = list(raw_feature_score.feature[0:500])
raw_train_x = pd.read_csv("../data/train_x.csv")[['uid']+fs]
raw_train_y = pd.read_csv("../data/train_y.csv")
raw_train = pd.merge(raw_train_x,raw_train_y,on='uid')
del raw_train_x,raw_train_y

raw_test = pd.read_csv("../data/test_x.csv")[['uid']+fs]
raw_train_unlabel = pd.read_csv('../data/train_unlabeled.csv')[['uid']+fs]

#将原始特征，排序特征，离散特征，以及其他11维特征（n1～n10，discret_null）合并
train = pd.merge(raw_train,rank_train,on='uid')
train = pd.merge(train,discret_train,on='uid')
train = pd.merge(train,train_eleven,on='uid')

test = pd.merge(raw_test,rank_test,on='uid')
test = pd.merge(test,discret_test,on='uid')
test = pd.merge(test,test_eleven,on='uid')
test_uid = test.uid



#从无标签数据里面选取部分样本作为负样本
xgb717_predict_unlabeled_data = pd.read_csv('./xgb717_predict_unlabeled_data.csv')
unlabeldata_0 = xgb717_predict_unlabeled_data[xgb717_predict_unlabeled_data.score<0.16]  #2672个

tmp = pd.merge(unlabeldata_0,raw_train_unlabel,on="uid",how="left")
tmp1 = pd.merge(tmp,rank_train_unlabeled,on="uid",how="left")
tmp2 = pd.merge(tmp1,trainunlabeled_eleven,on="uid",how="left")
neg_sample = pd.merge(tmp2,discret_train_unlabeled,on="uid",how="left")
neg_sample = neg_sample.drop(["score","uid"],axis=1)
neg_sample['y'] = [0 for _ in range(len(neg_sample))]

print "select {0} negative sample from train_unlabel.csv".format(len(newdata0))
del unlabeldata_0,tmp,tmp1,tmp2



def pipeline(iteration,random_seed,feature_num,rank_feature_num,discret_feature_num,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight):
    #选取的前n个原始特征、排序特征、离散特征
    raw_feature_selected = list(raw_feature_score.feature[0:feature_num])
    rank_feature_selected = list(rank_feature_score.feature[0:rank_feature_num])
    discret_feature_selected = list(discret_feature_score.feature[0:discret_feature_num])

    #根据选取的特征构造出训练集，测试集，以及从无标签数据中获取的负样本
    train_xy = train[eleven_feature+feature_selected+rank_feature_selected+discret_feature_selected+['y']]
    train_xy[train_xy<0] = -1    #缺失值-1或-2，都统一对待，设置为-1

    test_x = test[eleven_feature+feature_selected+rank_feature_selected+discret_feature_selected]
    test_x[test_x<0] = -1

    neg = neg_sample[eleven_feature+feature_selected+rank_feature_selected+discret_feature_selected+['y']]
    neg[neg<0] = -1   
    
    #将从无标签数据中选取出的负样本和原始训练数据合并
    train_xy = pd.concat([train_xy,neg])
    y = train_xy.y
    X = train_xy.drop(['y'],axis=1)
    
    #xgboost start
    dtest = xgb.DMatrix(test_x)
    dtrain = xgb.DMatrix(X, label=y)
    params={
    	'booster':'gbtree',
    	'objective': 'binary:logistic',
    	'scale_pos_weight': float(len(y)-sum(y))/float(sum(y)),
        'eval_metric': 'auc',
    	'gamma':gamma,
    	'max_depth':max_depth,
    	'lambda':lambd,
        'subsample':subsample,
        'colsample_bytree':colsample_bytree,
        'min_child_weight':min_child_weight, 
        'eta': 0.08,
    	'seed':random_seed,
    	'nthread':8
        }
    
    watchlist  = [(dtrain,'train')]
    model = xgb.train(params,dtrain,num_boost_round=1024,evals=watchlist)
    model.save_model('./model/xgb{0}.model'.format(iteration))
    
    #predict test set
    test_y = model.predict(dtest)
    test_result = pd.DataFrame(test_uid,columns=["uid"])
    test_result["score"] = test_y
    test_result.to_csv("./preds/xgb{0}.csv".format(iteration),index=None,encoding='utf-8')
    
    #save feature score
    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
    fs = []
    for (key,value) in feature_score:
        fs.append("{0},{1}\n".format(key,value))
    
    with open('./featurescore/feature_score_{0}.csv'.format(iteration),'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)



if __name__ == "__main__":
    """
    random_seed = range(1000,2000,10)
    feature_num = range(300,500,2)
    rank_feature_num = range(300,500,2)
    discret_feature_num = range(64,100,1)
    gamma = [i/1000.0 for i in range(0,300,3)]
    max_depth = [6,7,8]
    lambd = range(500,700,2)
    subsample = [i/1000.0 for i in range(500,700,2)]
    colsample_bytree = [i/1000.0 for i in range(250,350,1)]
    min_child_weight = [i/1000.0 for i in range(250,550,3)]
    random.shuffle(rank_feature_num)
    random.shuffle(random_seed)
    random.shuffle(feature_num)
    random.shuffle(discret_feature_num)
    random.shuffle(gamma)
    random.shuffle(max_depth)
    random.shuffle(lambd)
    random.shuffle(subsample)
    random.shuffle(colsample_bytree)
    random.shuffle(min_child_weight)
    
    with open('params.pkl','w') as f:
        cPickle.dump((random_seed,feature_num,rank_feature_num,discret_feature_num,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight),f)
    """

    with open('params_for_reproducing.pkl','r') as f:
        random_seed,feature_num,rank_feature_num,discret_feature_num,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight = cPickle.load(f)
    
    
    for i in range(36):
        print "iter:",i
        pipeline(i,random_seed[i],feature_num[i],rank_feature_num[i],discret_feature_num[i],gamma[i],max_depth[i%3],lambd[i],subsample[i],colsample_bytree[i],min_child_weight[i])
