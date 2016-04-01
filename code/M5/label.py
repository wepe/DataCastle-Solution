#coding=utf-8

"""
给无标签数据打标签，每10个样本有1024种标签组合，运行1024次xgboost，选取auc提升最大的组合。

"""

from sklearn.cross_validation import train_test_split
from sklearn import metrics
import pandas as pd
import xgboost as xgb
import sys,random,cPickle



train_xy = pd.read_csv('train_xy.csv')
train,val = train_test_split(train_xy,test_size=0.85,random_state=1024)

val_y = val.y
val_X = val.drop(['y'],axis=1)
dval = xgb.DMatrix(val_X)


train_unlabel = pd.read_csv('train_unlabeled.csv')


def pipeline(unlabel_data):
    """
    unlabel_data:
       columns=['uid','y',features]
    """
    this_train = pd.concat([train,unlabel_data])
    y = this_train.y
    X = this_train.drop(['y'],axis=1)
    dtrain = xgb.DMatrix(X, label=y)
    params={
    	'booster':'gbtree',
    	'objective': 'rank:pairwise',
    	'scale_pos_weight': float(len(y)-sum(y))/float(sum(y)),
        'eval_metric': 'auc',
    	'gamma':0.1,
    	'max_depth':8,
    	'lambda':600,
        'subsample':0.6,
        'colsample_bytree':0.3,
        'min_child_weight':0.3, 
        'eta': 0.04,
    	'seed':1024,
    	'nthread':20
        }
    model = xgb.train(params,dtrain,num_boost_round=256,verbose_eval=False)
    val_y_pred = model.predict(dval)
    fpr,tpr,thresholds = metrics.roc_curve(val_y,val_y_pred,pos_label=1)
    return metrics.auc(fpr,tpr)


labels = []
get_bin = lambda x: format(x, 'b').zfill(10)
for i in range(1024):
    label_str = get_bin(i)
    label = []
    for c in label_str:
        label.append(int(c))
    labels.append(label)


for i in range(5000):
    uid_index = range(i*10,(i+1)*10)
    samples_selected = train_unlabel.loc[uid_index]
    best_auc = 0
    best_label = []
    for label in labels:
        samples_selected['y'] = label
        this_auc = pipeline(samples_selected)
        print this_auc
        if this_auc>best_auc:
            best_auc = this_auc
            best_label = label
    
    with open('label.csv','a') as f:
        f.writelines(str(i)+','+','.join([str(i) for i in best_label])+','+str(best_auc)+'\n')
    
