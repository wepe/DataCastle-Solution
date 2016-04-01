#coding=utf-8

from sklearn.cross_validation import train_test_split
import pandas as pd
import xgboost as xgb
import sys,random
import cPickle



#离散特征,特征名称与rank特征重了，需要重命名，统一在之前加‘d’
discret_feature_score = pd.read_csv('./discret_feature_score.csv')
fs = list(discret_feature_score.feature[0:500])
discret_train = pd.read_csv("../data/train_x_discretization.csv")
discret_test = pd.read_csv("../data/test_x_discretization.csv")
discret_train_unlabeled = pd.read_csv("../data/train_unlabeled_discretization.csv")


#discret_null feature
test_dnull = pd.read_csv('../data/test_x_null.csv')[['uid','discret_null']]
train_dnull = pd.read_csv('../data/train_x_null.csv')[['uid','discret_null']]
trainunlabeled_dnull = pd.read_csv('../data/train_unlabeled_null.csv')[['uid','discret_null']]

#n_discret feature
test_nd = pd.read_csv('../data/test_x_nd.csv')[['uid','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10']]
train_nd = pd.read_csv('../data/train_x_nd.csv')[['uid','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10']]
trainunlabeled_nd = pd.read_csv('../data/train_unlabeled_nd.csv')[['uid','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10']]

#
discret_feature = ['n1','n2','n3','n4','n5','n6','n7','n8','n9','n10','discret_null']
test_d = pd.merge(test_nd,test_dnull,on='uid')
train_d = pd.merge(train_nd,train_dnull,on='uid')
trainunlabeled_d = pd.merge(trainunlabeled_nd,trainunlabeled_dnull,on='uid')

del test_dnull,train_dnull,trainunlabeled_dnull
del test_nd,train_nd,trainunlabeled_nd

#rank_feature
rank_feature_score = pd.read_csv('./rank_feature_score.csv')
fs = list(rank_feature_score.feature[0:500])
#load data
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

#raw data
feature_score_717 = pd.read_csv('./raw_feature_score.csv')
fs = list(feature_score_717.feature[0:500])
train_x = pd.read_csv("../data/train_x.csv")[['uid']+fs]
train_y = pd.read_csv("../data/train_y.csv")
train_xy = pd.merge(train_x,train_y,on='uid')
del train_x,train_y
train = pd.merge(train_xy,rank_train,on='uid')
train = pd.merge(train,train_d,on='uid')
train = pd.merge(train,discret_train,on='uid')

test = pd.read_csv("../data/test_x.csv")[['uid']+fs]
test = pd.merge(test,rank_test,on='uid')
test = pd.merge(test,test_d,on='uid')
test = pd.merge(test,discret_test,on='uid')
test_uid = test.uid

train_unlabel = pd.read_csv('../data/train_unlabeled.csv')[['uid']+fs]
tmp1 = pd.merge(train_unlabel,rank_train_unlabeled,on="uid",how="left")
tmp2 = pd.merge(tmp1,trainunlabeled_d,on="uid",how="left")
newdata = pd.merge(tmp2,discret_train_unlabeled,on="uid",how="left")
newdata[newdata<0] = -1
print "select {0} sample from train_unlabel.csv".format(len(newdata))


feature_selected = list(feature_score_717.feature[0:500])
rank_feature_selected = list(rank_feature_score.feature[0:500])
discret_feature_selected = list(discret_feature_score.feature[0:100])

train_xy = train[['uid']+discret_feature+feature_selected+rank_feature_selected+discret_feature_selected+['y']]
train_xy[train_xy<0] = -1
train_xy.to_csv('train_xy.csv',index=None)

n = newdata[['uid']+discret_feature+feature_selected+rank_feature_selected+discret_feature_selected]
n.to_csv('train_unlabeled.csv',index=None)
