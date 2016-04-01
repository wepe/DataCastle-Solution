#coding=utf-8

"""
对label.csv里的数据按照bset_auc排序，然后选取top5000个无标签样本（即线下auc提升最大的无标签样本）
每次从这top5000个样本里随机选取50个样本
"""
import pandas as pd
import random,os

labels = pd.read_csv('label.csv',header=None)
labels.columns = ['ind','sample1','sample2','sample3','sample4','sample5','sample6','sample7','sample8','sample9','sample10','auc']
#对auc排序
labels['rk'] = labels.auc.rank(ascending=False)
#选取top5000的样本，注意每组10个样本，top5000即选取rank值小于500的组合
labels = labels[labels.rk<=500]

#每次随机取50个样本，即随机取5组样本
#相当于随机打乱，生成一百份文件
inds = list(labels.ind)
random.shuffle(inds)

os.mkdir('samples_selected')
for i in range(100):
    y = ['sample1','sample2','sample3','sample4','sample5','sample6','sample7','sample8','sample9','sample10']
    five_inds = inds[(5*i):(5*(i+1))]
    five_inds_label = [labels[labels.ind==this_ind][y].values.tolist()  for this_ind in five_inds]
    
    five_inds_label_ = []
    [five_inds_label_.extend(j[0]) for j in five_inds_label]

    temp = [range(ind*10,(ind+1)*10) for ind in five_inds]
    uid_index = []
    [uid_index.extend(t) for t in temp]

    
    #无标签样本的uid
    train_unlabel_uid = pd.read_csv('train_unlabeled.csv')
    sample50 = train_unlabel_uid.loc[uid_index]
    sample50['y'] = five_inds_label_
    sample50[['uid','y']].to_csv('samples_selected/{0}.csv'.format(i),index=None)
