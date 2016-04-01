#coding=utf-8

"""
Author: wepon (http://2hwp.com/)
Code: https://github.com/wepe/DataCastle-Solution

"""

import pandas as pd

train = pd.read_csv("../data/train_x_rank.csv")
train_x = train.drop(['uid'],axis=1)
test = pd.read_csv("../data/test_x_rank.csv")
test_x = test.drop(['uid'],axis=1)
train_unlabeled =  pd.read_csv("../data/train_unlabeled_rank.csv")
train_unlabeled_x =  train_unlabeled.drop(['uid'],axis=1)


train_x[train_x<1500] = 1
train_x[(train_x>=1500)&(train_x<3000)] = 2
train_x[(train_x>=3000)&(train_x<4500)] = 3
train_x[(train_x>=4500)&(train_x<6000)] = 4
train_x[(train_x>=6000)&(train_x<7500)] = 5
train_x[(train_x>=7500)&(train_x<9000)] = 6
train_x[(train_x>=9000)&(train_x<10500)] = 7
train_x[(train_x>=10500)&(train_x<12000)] = 8
train_x[(train_x>=12000)&(train_x<13500)] = 9
train_x[train_x>=13500] = 10
#离散特征的命名：在原始特征前加'd',如'x1'的离散特征为'dx1'
rename_dict = {s:'d'+s[1:] for s in train_x.columns.tolist()}
train_x = train_x.rename(columns=rename_dict)
train_x['uid'] = train.uid
train_x.to_csv('../data/train_x_discretization.csv',index=None)

train_unlabeled_x[train_unlabeled_x<5000] = 1
train_unlabeled_x[(train_unlabeled_x>=5000)&(train_unlabeled_x<10000)] = 2
train_unlabeled_x[(train_unlabeled_x>=10000)&(train_unlabeled_x<15000)] = 3
train_unlabeled_x[(train_unlabeled_x>=15000)&(train_unlabeled_x<20000)] = 4
train_unlabeled_x[(train_unlabeled_x>=20000)&(train_unlabeled_x<25000)] = 5
train_unlabeled_x[(train_unlabeled_x>=25000)&(train_unlabeled_x<30000)] = 6
train_unlabeled_x[(train_unlabeled_x>=30000)&(train_unlabeled_x<35000)] = 7
train_unlabeled_x[(train_unlabeled_x>=35000)&(train_unlabeled_x<40000)] = 8
train_unlabeled_x[(train_unlabeled_x>=40000)&(train_unlabeled_x<45000)] = 9
train_unlabeled_x[train_unlabeled_x>=45000] = 10
train_unlabeled_x = train_unlabeled_x.rename(columns=rename_dict)
train_unlabeled_x['uid'] = train_unlabeled.uid
train_unlabeled_x.to_csv('../data/train_unlabeled_discretization.csv',index=None)

test_x[test_x<500] = 1
test_x[(test_x>=500)&(test_x<1000)] = 2
test_x[(test_x>=1000)&(test_x<1500)] = 3
test_x[(test_x>=1500)&(test_x<2000)] = 4
test_x[(test_x>=2000)&(test_x<2500)] = 5
test_x[(test_x>=2500)&(test_x<3000)] = 6
test_x[(test_x>=3000)&(test_x<3500)] = 7
test_x[(test_x>=3500)&(test_x<4000)] = 8
test_x[(test_x>=4000)&(test_x<4500)] = 9
test_x[test_x>=4500] = 10
test_x = test_x.rename(columns=rename_dict)
test_x['uid'] = test.uid
test_x.to_csv('../data/test_x_discretization.csv',index=None)

