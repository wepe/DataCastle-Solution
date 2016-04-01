#coding=utf-8

"""
Author: wepon (http://2hwp.com/)
Code: https://github.com/wepe/DataCastle-Solution

"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

train = pd.read_csv('../data/train_x_null.csv')[['uid','n_null']]
train_y = pd.read_csv('../data/train_y.csv')
train = pd.merge(train,train_y,on='uid')
train = train.sort(columns='n_null')

train_unlabel = pd.read_csv('../data/train_unlabeled_null.csv')[['uid','n_null']]
train_unlabel = train_unlabel.sort(columns='n_null')

test = pd.read_csv('../data/test_x_null.csv')[['uid','n_null']]
test = test.sort(columns='n_null')

t = train.n_null.values
y = train.y.values
y0 = [ i+1-sum(y[0:i+1]) for i in range(len(y))]
x = range(len(t))
plt.scatter(x,t,c='k')
plt.plot(x,y0,c='b')
plt.title('train set')
plt.show()


t = train_unlabel.n_null.values
x = range(len(t))
plt.scatter(x,t,c='k')
plt.title('unlabeled set')
plt.show()


t = test.n_null.values
x = range(len(t))
plt.scatter(x,t,c='k')
plt.title('test set')
plt.show()


