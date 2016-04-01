#coding=utf-8

"""
Author: wepon (http://2hwp.com/)
Code: https://github.com/wepe/DataCastle-Solution

"""

import pandas as pd


train_x = pd.read_csv("../data/train_x.csv")
test_x = pd.read_csv("../data/test_x.csv")
train_unlabeled =  pd.read_csv("../data/train_unlabeled.csv")

train_x['n_null'] = (train_x<0).sum(axis=1)
test_x['n_null'] = (test_x<0).sum(axis=1)
train_unlabeled['n_null'] = (train_unlabeled<0).sum(axis=1)

train_x['discret_null'] = train_x.n_null
train_x.discret_null[train_x.discret_null<=32] = 1
train_x.discret_null[(train_x.discret_null>32)&(train_x.discret_null<=69)] = 2
train_x.discret_null[(train_x.discret_null>69)&(train_x.discret_null<=147)] = 3
train_x.discret_null[(train_x.discret_null>147)&(train_x.discret_null<=194)] = 4
train_x.discret_null[(train_x.discret_null>194)] = 5
train_x[['uid','n_null','discret_null']].to_csv('../data/train_x_null.csv',index=None)

test_x['discret_null'] = test_x.n_null
test_x.discret_null[test_x.discret_null<=32] = 1
test_x.discret_null[(test_x.discret_null>32)&(test_x.discret_null<=69)] = 2
test_x.discret_null[(test_x.discret_null>69)&(test_x.discret_null<=147)] = 3
test_x.discret_null[(test_x.discret_null>147)&(test_x.discret_null<=194)] = 4
test_x.discret_null[(test_x.discret_null>194)] = 5
test_x[['uid','n_null','discret_null']].to_csv('../data/test_x_null.csv',index=None)

train_unlabeled['discret_null'] = train_unlabeled.n_null
train_unlabeled.discret_null[train_unlabeled.discret_null<=32] = 1
train_unlabeled.discret_null[(train_unlabeled.discret_null>32)&(train_unlabeled.discret_null<=69)] = 2
train_unlabeled.discret_null[(train_unlabeled.discret_null>69)&(train_unlabeled.discret_null<=147)] = 3
train_unlabeled.discret_null[(train_unlabeled.discret_null>147)&(train_unlabeled.discret_null<=194)] = 4
train_unlabeled.discret_null[(train_unlabeled.discret_null>194)] = 5
train_unlabeled[['uid','n_null','discret_null']].to_csv('../data/train_unlabeled_null.csv',index=None)
