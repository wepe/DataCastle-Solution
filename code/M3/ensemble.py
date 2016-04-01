#coding=utf-8

"""
加权融合代码示例
"""

import pandas as pd


xgb717 = pd.read_csv("xgb717.csv")
svm6938 = pd.read_csv('svm6938.csv')
xgb725 = pd.read_csv('725.csv')

uid = xgb717.uid
score = 0.15*xgb717.score+0.25*svm6938.score+0.6*xgb725.score
pred = pd.DataFrame(uid,columns=['uid'])
pred['score'] = score

pred.to_csv('submission.csv',index=None,encoding='utf-8')





