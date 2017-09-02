#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from collections import defaultdict 
import random
#from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score


class Bagging(object):
    
    def __init__(self,n_estimators,estimator):
        self.estimator = estimator
        self.n_estimators = n_estimators

        
    def Voting(self,data):
        #投票法
        term = np.transpose(data)    #转置
        result =list()              #存储结果
        
        def Vote(df):               #对每一行做投票
            store = defaultdict()
            for kw in df:
                store.setdefault(kw, 0)
                store[kw] += 1
            return max(store,key=store.get)
        
        result= map(Vote,term)     #获取结果
        return result
        
    def SubSample(self,n,data):
        pass
    
    def TrainPredict(self,train,test):
        clf = self.estimator.fit(train[:,0:-1],train[:,-1])
        result = clf.predict(test[:,0:-1])
        return result
    
    def RepetitionRandomSampling(self,data,number):     #有放回采样，number为抽样的个数
        sample=[]
        for i in range(number):
             sample.append(data[random.randint(0,len(data)-1)])
        return sample
    
    def Bagging_clf(self,train,test):
        result = list()
        for i in range(self.n_estimators):
            #print i
            sample=self.RepetitionRandomSampling(train,len(train))        #构建数据集
            #print sample
            result.append(self.TrainPredict(np.array(sample),np.array(test)))    #训练模型 返回每个模型的输出
            #print result
        score = self.Voting(result) 
        recall=recall_score(test[:,-1], score, average=None) 
        print "自己的Bagging"
        return recall
        #print recall
    


if __name__ == "__main__":
    from sklearn import tree
    from sklearn.ensemble import BaggingClassifier
    datafile = "../data/Yeast.data"
    data =pd.read_csv(datafile)
    data_x = data.iloc[:,0:-1]
    data_y = data.iloc[:,-1]
    

    x_train, x_test, y_train, y_test = train_test_split(data_x,data_y,test_size=0.33,random_state=120)
    train = np.column_stack([x_train, y_train])
    test = np.column_stack([x_test, y_test])
    
    clf = tree.DecisionTreeClassifier()
    clf_self = Bagging(n_estimators = 200,estimator = clf)
    recall_self = clf_self.Bagging_clf(train,test)
    #print "自己的Bagging"
    print recall_self 
    
    
    clfba = BaggingClassifier(base_estimator=clf,n_estimators=200)
    clfba.fit(x_train, y_train)
    re = clfba.predict(x_test)
    recall=recall_score(y_test, re, average=None) 
    print "库函数bagging结果"
    print recall
    
    #mm= qq.RepetitionRandomSampling([11,11,22],910)


