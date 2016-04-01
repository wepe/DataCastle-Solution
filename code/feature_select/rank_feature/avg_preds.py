#coding=utf-8

"""
Author: wepon (http://2hwp.com/)
Code: https://github.com/wepe/DataCastle-Solution

"""

import pandas as pd 
import os


files = os.listdir('./preds')
pred = pd.read_csv('./preds/'+files[0])
uid = pred.uid
score = pred.score
for f in files[1:]:
    pred = pd.read_csv('./preds/'+f)
    score += pred.score

score /= len(files)

pred = pd.DataFrame(uid,columns=['uid'])
pred['score'] = score
pred.to_csv('avg_preds.csv',index=None,encoding='utf-8')
