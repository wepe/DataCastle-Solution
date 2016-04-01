#coding=utf-8

"""
Author: wepon (http://2hwp.com/)
Code: https://github.com/wepe/DataCastle-Solution

"""

import pandas as pd

train_x = pd.read_csv('../data/train_x_discretization.csv')
test_x = pd.read_csv('../data/test_x_discretization.csv')
train_unlabeled_x =  pd.read_csv('../data/train_unlabeled_discretization.csv')

train_x['n1'] = (train_x==1).sum(axis=1)
train_x['n2'] = (train_x==2).sum(axis=1)
train_x['n3'] = (train_x==3).sum(axis=1)
train_x['n4'] = (train_x==4).sum(axis=1)
train_x['n5'] = (train_x==5).sum(axis=1)
train_x['n6'] = (train_x==6).sum(axis=1)
train_x['n7'] = (train_x==7).sum(axis=1)
train_x['n8'] = (train_x==8).sum(axis=1)
train_x['n9'] = (train_x==9).sum(axis=1)
train_x['n10'] = (train_x==10).sum(axis=1)
train_x[['uid','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10']].to_csv('../data/train_x_nd.csv',index=None)

test_x['n1'] = (test_x==1).sum(axis=1)
test_x['n2'] = (test_x==2).sum(axis=1)
test_x['n3'] = (test_x==3).sum(axis=1)
test_x['n4'] = (test_x==4).sum(axis=1)
test_x['n5'] = (test_x==5).sum(axis=1)
test_x['n6'] = (test_x==6).sum(axis=1)
test_x['n7'] = (test_x==7).sum(axis=1)
test_x['n8'] = (test_x==8).sum(axis=1)
test_x['n9'] = (test_x==9).sum(axis=1)
test_x['n10'] = (test_x==10).sum(axis=1)
test_x[['uid','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10']].to_csv('../data/test_x_nd.csv',index=None)

train_unlabeled_x['n1'] = (train_unlabeled_x==1).sum(axis=1)
train_unlabeled_x['n2'] = (train_unlabeled_x==2).sum(axis=1)
train_unlabeled_x['n3'] = (train_unlabeled_x==3).sum(axis=1)
train_unlabeled_x['n4'] = (train_unlabeled_x==4).sum(axis=1)
train_unlabeled_x['n5'] = (train_unlabeled_x==5).sum(axis=1)
train_unlabeled_x['n6'] = (train_unlabeled_x==6).sum(axis=1)
train_unlabeled_x['n7'] = (train_unlabeled_x==7).sum(axis=1)
train_unlabeled_x['n8'] = (train_unlabeled_x==8).sum(axis=1)
train_unlabeled_x['n9'] = (train_unlabeled_x==9).sum(axis=1)
train_unlabeled_x['n10'] = (train_unlabeled_x==10).sum(axis=1)
train_unlabeled_x[['uid','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10']].to_csv('../data/train_unlabeled_nd.csv',index=None)
