#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict 
import random
from sklearn.ensemble import AdaBoostClassifier
#from sklearn.model_selection import train_test_split

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
            print i
            sample=self.RepetitionRandomSampling(data,len(data))        #构建数据集
            #print sample
            
            result.append(self.TrainPredict(np.array(train),np.array(test)))    #训练模型 返回每个模型的输出
            #print result
        score = self.Voting(result) 
        print score
    


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    iris = load_iris()
    data = iris.data
    target = iris.target
    term = np.column_stack([data,target])
#    term1 = [1,2,3,4,2,3,1,2,3,2,2,2,2,2,1]
#    term2 =[3,1,2,4,2,3,1,2,3,1,1,2,2,2,2]
#    term3 =[3,1,2,4,2,3,1,2,3,1,1,2,2,2,1]
#    term4 =[2,1,2,4,2,3,3,1,2,2,1,3,2,1,2]
#    term5 =[2,1,2,1,2,3,2,2,3,1,1,2,2,3,3]
#    
#    term = np.array([term1,term2,term3,term4,term5])
    clf = AdaBoostClassifier(n_estimators= 10,learning_rate=0.01)
   
    qq = Bagging(n_estimators = 10,estimator = clf)
   # print qq.TrainPredict(term)
    train = term[0:int(0.9*len(term)),:]
    test = term[int(0.9*len(term)):,:]
    score = qq.Bagging_clf(train,test)
    
    #mm= qq.RepetitionRandomSampling([11,11,22],910)


