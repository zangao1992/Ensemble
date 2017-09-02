#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2017-08-28 

@author: panda_zjd
"""
import numpy as np
import pandas as pd
from collections import defaultdict 
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import IsolationForest

class Bagging(object):
    
    def __init__(self,n_estimators,estimator,rate=1.0):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.rate = rate

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

    #随机欠采样函数
    def UnderSampling(self,data):
        #np.random.seed(np.random.randint(0,1000))
        data=np.array(data)
        np.random.shuffle(data)    #打乱data          
        newdata = data[0:int(data.shape[0]*self.rate),:]   #切片，取总数*rata的个数，删去（1-rate）%的样本
        return newdata   
    
    #isolationforest 欠采样
    def IF_SubSample(self,data,num):
        clf_IF = IsolationForest(n_estimators=200,contamination=(1.0-self.rate))
        clf_IF.fit(data)
        pred =clf_IF.predict(data)
        score = clf_IF.decision_function(data)

    def TrainPredict(self,train,test):          #训练基础模型，并返回模型预测结果
        clf = self.estimator.fit(train[:,0:-1],train[:,-1])
        result = clf.predict(test[:,0:-1])
        return result
    
    #简单有放回采样
    def RepetitionRandomSampling(self,data,number):     #有放回采样，number为抽样的个数
        sample=[]
        for i in range(int(self.rate*number)):
             sample.append(data[random.randint(0,len(data)-1)])
        return sample
    
    def Metrics(self,predict_data,test):        #评价函数
        score = predict_data
        recall=recall_score(test[:,-1], score, average=None)    #召回率
        precision=precision_score(test[:,-1], score, average=None)  #查准率
        return recall,precision
    '''    
    def Bagging_clf(self,train,test,sample_type = "RepetitionRandomSampling"):
        print "self.Bagging single_basemodel"
        result = list()
        
        if sample_type == "RepetitionRandomSampling":
            print "选择的采样方法：",sample_type
            sample_function = self.RepetitionRandomSampling
        elif sample_type == "UnderSampling":
            print "选择的采样方法：",sample_type
            sample_function = self.UnderSampling 
            print "采样率",self.rate
        elif sample_type == "IF_SubSample":
            print "选择的采样方法：",sample_type
            sample_function = self.IF_SubSample 
            print "采样率",(1.0-self.rate)
        print sample_function(train,len(train))
        for i in range(self.n_estimators):
            sample=sample_function(train,len(train))        #构建数据集
            print sample
            result.append(self.TrainPredict(np.array(sample),np.array(test)))    #训练模型 返回每个模型的输出
        print result
        score = self.Voting(result) 
        recall,precosoion = self.Metrics(score,test)
        return recall,precosoion                                         
    '''   
    def MutModel_clf(self,train,test,sample_type = "RepetitionRandomSampling"):
        print "self.Bagging Mul_basemodel"
        result = list()
        num_estimators =len(self.estimator)   #使用基础模型的数量

        if sample_type == "RepetitionRandomSampling":
            print "选择的采样方法：",sample_type
            sample_function = self.RepetitionRandomSampling
        elif sample_type == "UnderSampling":
            print "选择的采样方法：",sample_type
            sample_function = self.UnderSampling 
            print "采样率",self.rate
        elif sample_type == "IF_SubSample":
            print "选择的采样方法：",sample_type
            sample_function = self.IF_SubSample 
            print "采样率",(1.0-self.rate)
            
        for estimator in self.estimator:
            print estimator
            for i in range(int(self.n_estimators/num_estimators)):
                sample=np.array(sample_function(train,len(train)))       #构建数据集
                clf = estimator.fit(sample[:,0:-1],sample[:,-1])
                result.append(clf.predict(test[:,0:-1]))      #训练模型 返回每个模型的输出
        
        score = self.Voting(result)
        recall,precosoion = self.Metrics(score,test)
        return recall,precosoion    

if __name__ == "__main__":
    from sklearn import tree
    from sklearn.ensemble import BaggingClassifier
    from sklearn.ensemble import AdaBoostClassifier
    
    datafile = "../data/waveform.data"
    data = pd.read_csv(datafile)
    data_x = data.iloc[:,0:-1]
    data_y = data.iloc[:,-1]
    
    x_train, x_test, y_train, y_test = train_test_split(data_x,data_y,test_size=0.33,random_state=120)
    train = np.column_stack([x_train, y_train])
    test = np.column_stack([x_test, y_test])
    
    clf = [tree.DecisionTreeClassifier(),AdaBoostClassifier(),tree.DecisionTreeClassifier(max_depth=4)]     #基础模型
    #clf = [tree.DecisionTreeClassifier()]
    clf_self = Bagging(n_estimators = 200, estimator = clf,rate =1.0)
    if(len(clf_self.estimator) == 1):
        print "bagging只有一个基础模型"
        #recall_self,precision_self = clf_self.MutModel_clf(train,test)   #单基础模型
    elif(len(clf_self.estimator)>1):
        print "bagging有多个基础模型"
       # recall_self,precision_self = clf_self.MutModel_clf(train,test)   #多基础模型
    else:
        print "请输出基础模型"
    recall_self,precision_self = clf_self.MutModel_clf(train,test)
    print "recall:",'\n',recall_self
    print "precision",'\n',precision_self 
    
    #sklearn中 BaggingClassifier
    clf_sklearn = BaggingClassifier(base_estimator = tree.DecisionTreeClassifier(),n_estimators=200)
    clf_sklearn.fit(x_train, y_train)
    score = clf_sklearn.predict(x_test)
    recall=recall_score(y_test, score, average=None) 
    precision=precision_score(y_test, score, average=None) 
    print "*******"*10
    print "sklern.bagging结果"
    print "recall:",'\n',recall
    print "precision",'\n',precision
    
    print "*******"*10
    print 
    print recall_self-recall
    print precision_self-precision