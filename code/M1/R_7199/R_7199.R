library(xgboost)
setwd('G:\\比赛\\微额借款用户人品预测')
# read data
train=read.csv('train_x.csv')
test=read.csv('test_x.csv')
train.y=read.csv('train_y.csv')
ft=read.csv('features_type.csv')
fn.cat=as.character(ft[ft[,2]=='category',1])

fn.num=as.character(ft[ft[,2]=='numeric',1])


# create dummy variables
temp.train=data.frame(rep(0,nrow(train)))
temp.test=data.frame(rep(0,nrow(test)))
for(f in fn.cat){
  levels=unique(train[,f])
  col.train=data.frame(factor(train[,f],levels=levels))
  col.test=data.frame(factor(test[,f],levels=levels))
  colnames(col.train)=f
  colnames(col.test)=f
  temp.train=cbind(temp.train,model.matrix(as.formula(paste0('~',f,'-1')),data=col.train))
  temp.train[,paste0(f,'-1')]=NULL
  temp.test=cbind(temp.test,model.matrix(as.formula(paste0('~',f,'-1')),data=col.test))
  temp.test[,paste0(f,'-1')]=NULL
}
temp.train[,1]=NULL
temp.test[,1]=NULL
train.new=as.matrix(data.matrix(cbind(train[,c('uid',fn.num)],temp.train)),sparse=T)
test.new=as.matrix(data.matrix(cbind(test[,c('uid',fn.num)],temp.test)),sparse=T)


# fit xgboost model

dtrain=xgb.DMatrix(data=train.new[,-1],label=1-train.y$y)
dtest= xgb.DMatrix(data=test.new[,-1])

model=xgb.train(booster='gbtree',
                objective='binary:logistic',
                scale_pos_weight=8.7,
                gamma=0,
                lambda=700,
                subsample=0.7,
                colsample_bytree=0.30,
                min_child_weight=5,
                max_depth=8,
                eta=0.02,
                data=dtrain,
                nrounds=1520,
                metrics='auc',
                nthread=8)

# predict probabilities
pred=1-predict(model,dtest)

write.csv(data.frame('uid'=test.new[,1],'score'=pred),file='R_7199.csv',row.names=F)