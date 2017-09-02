#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from collections import defaultdict 
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

class Bagging(object):
    
    def __init__(self,n_estimators,estimator):
        self.estimator = estimator
        self.n_estimators = n_estimators

    def Voting(self,data):          #投票法
        term = np.transpose(data)   #转置
        result =list()              #存储结果
        
        def Vote(df):               #对每一行做投票
            store = defaultdict()
            for kw in df:
                store.setdefault(kw, 0)
                store[kw] += 1
            return max(store,key=store.get)
        
        result= map(Vote,term)      #获取结果
        return result
        
    def SubSample(self,n,data):
        pass
    
    def TrainPredict(self,train,test):          #训练基础模型，并返回模型预测结果
        clf = self.estimator.fit(train[:,0:-1],train[:,-1])
        result = clf.predict(test[:,0:-1])
        return result
    
    def RepetitionRandomSampling(self,data,number):     #有放回采样，number为抽样的个数
        sample=[]
        for i in range(number):
             sample.append(data[random.randint(0,len(data)-1)])
        return sample
    
    def Metrics(self,predict_data):
        #评价函数
        recall=recall_score(test[:,-1], score, average=None)    #召回率
        precision=precision_score(y_test, score, average=None)  #查准率
        return recall,precision
    
    def Bagging_clf(self,train,test):
        print "self.Bagging single_basemodel"
        result = list()
        for i in range(self.n_estimators):
            sample=self.RepetitionRandomSampling(train,len(train))        #构建数据集
            result.append(self.TrainPredict(np.array(sample),np.array(test)))    #训练模型 返回每个模型的输出
        score = self.Voting(result) 
        recall,precosoion = self.Metrics(score)
        return recall,precosoion                                         
    
    def MutModel_clf(self,train,test):
        print "self.Bagging Mul_basemodel"
        result = list()
        num_estimators =len(self.estimator)
        for estimator in self.estimator:
            print estimator
            for i in range(int(self.n_estimators/num_estimators)):
                sample=np.array(self.RepetitionRandomSampling(train,len(train)) )       #构建数据集
                clf = estimator.fit(sample[:,0:-1],sample[:,-1])
                result.append(clf.predict(test[:,0:-1]))      #训练模型 返回每个模型的输出
        score = self.Voting(result)
        recall,precosoion = self.Metrics(score)
        return recall,precosoion    

if __name__ == "__main__":
    from sklearn import tree
    from sklearn.ensemble import BaggingClassifier
    from sklearn.ensemble import AdaBoostClassifier
    
    datafile = "../data/Yeast.data"
    data = pd.read_csv(datafile)
    data_x = data.iloc[:,0:-1]
    data_y = data.iloc[:,-1]
    
    x_train, x_test, y_train, y_test = train_test_split(data_x,data_y,test_size=0.33,random_state=120)
    train = np.column_stack([x_train, y_train])
    test = np.column_stack([x_test, y_test])
    
    clf = [tree.DecisionTreeClassifier(),AdaBoostClassifier()]     #基础模型
    clf_self = Bagging(n_estimators = 200, estimator = clf)
    if(len(clf_self.estimator) == 1):
        recall_self,precision_self = clf_self.Bagging_clf(train,test)   #单基础模型
    elif(len(clf_self.estimator)>1):
        recall_self,precision_self = clf_self.MutModel_clf(train,test)   #多基础模型
    else:
        print "请输出基础模型"
    print "recall:",'\n',recall_self
    print "precision",'\n',precision_self 
     
    #sklearn中 BaggingClassifier
    clf_sklearn = BaggingClassifier(base_estimator = tree.DecisionTreeClassifier() , n_estimators=200)
    clf_sklearn.fit(x_train, y_train)
    score = clf_sklearn.predict(x_test)
    recall=recall_score(y_test, score, average=None) 
    precision=precision_score(y_test, score, average=None) 
    print "*******"*10
    print "sklern.bagging结果"
    print "recall:",'\n',recall
    print "precision",'\n',precision
    