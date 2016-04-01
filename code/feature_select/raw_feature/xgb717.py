"""
Author: wepon (http://2hwp.com)
"""
print(__doc__)

from sklearn.cross_validation import train_test_split
import pandas as pd
import xgboost as xgb

random_seed = 1225

#set data path
train_x_csv = "../../data/train_x.csv"
train_y_csv = "../../data/train_y.csv"
test_x_csv = "../../data/test_x.csv"
features_type_csv = "../../data/features_type.csv"

#load data
train_x = pd.read_csv(train_x_csv)
train_y = pd.read_csv(train_y_csv)
train_xy = pd.merge(train_x,train_y,on='uid')

test = pd.read_csv(test_x_csv)
test_uid = test.uid
test_x = test.drop(['uid'],axis=1)

#dictionary {feature:type}
features_type = pd.read_csv(features_type_csv)
features_type.index = features_type.feature
features_type = features_type.drop('feature',axis=1)
features_type = features_type.to_dict()['type']


feature_info = {}
features = list(train_x.columns)
features.remove('uid')

for feature in features:
    max_ = train_x[feature].max()
    min_ = train_x[feature].min()
    n_null = len(train_x[train_x[feature]<0])  #number of null
    n_gt1w = len(train_x[train_x[feature]>10000])  #greater than 10000
    feature_info[feature] = [min_,max_,n_null,n_gt1w]

#see how many neg/pos sample
print "neg:{0},pos:{1}".format(len(train_xy[train_xy.y==0]),len(train_xy[train_xy.y==1]))


#split train set,generate train,val,test set
train_xy = train_xy.drop(['uid'],axis=1)
train,val = train_test_split(train_xy, test_size = 0.2,random_state=1)#random_state is of big influence for val-auc
y = train.y
X = train.drop(['y'],axis=1)
val_y = val.y
val_X = val.drop(['y'],axis=1)



#xgboost start here
dtest = xgb.DMatrix(test_x)
dval = xgb.DMatrix(val_X,label=val_y)
dtrain = xgb.DMatrix(X, label=y)
params={
	'booster':'gbtree',
	'objective': 'binary:logistic',
	'early_stopping_rounds':100,
	'scale_pos_weight': 1500.0/13458.0,
        'eval_metric': 'auc',
	'gamma':0.1,#0.2 is ok
	'max_depth':8,
	'lambda':550,
        'subsample':0.7,
        'colsample_bytree':0.3,
        'min_child_weight':2.5, 
        'eta': 0.007,
	'seed':random_seed,
	'nthread':7
    }

watchlist  = [(dtrain,'train'),(dval,'val')]#The early stopping is based on last set in the evallist
model = xgb.train(params,dtrain,num_boost_round=50000,evals=watchlist)
model.save_model('./model/xgb.model')
print "best best_ntree_limit",model.best_ntree_limit   #did not save the best,why?

#predict test set (from the best iteration)
test_y = model.predict(dtest,ntree_limit=model.best_ntree_limit)
test_result = pd.DataFrame(columns=["uid","score"])
test_result.uid = test_uid
test_result.score = test_y
test_result.to_csv("xgb.csv",index=None,encoding='utf-8')  #remember to edit xgb.csv , add ""


#save feature score and feature information:  feature,score,min,max,n_null,n_gt1w
feature_score = model.get_fscore()
for key in feature_score:
    feature_score[key] = [feature_score[key]]+feature_info[key]+[features_type[key]]

feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
fs = []
for (key,value) in feature_score:
    fs.append("{0},{1},{2},{3},{4},{5},{6}\n".format(key,value[0],value[1],value[2],value[3],value[4],value[5]))

with open('feature_score.csv','w') as f:
    f.writelines("feature,score,min,max,n_null,n_gt1w\n")
    f.writelines(fs)




