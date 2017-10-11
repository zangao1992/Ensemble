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
from sklearn.metrics import log_loss
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier


if __name__ == "__main__":

    
    datafile = "../data/waveform.data"
    data = pd.read_csv(datafile)
    data_x = data.iloc[:,0:-1]
    data_y = data.iloc[:,-1]
    
    x_train, x_test, y_train, y_test = train_test_split(data_x,data_y,test_size=0.33,random_state=120)
    train = np.column_stack([x_train, y_train])
    test = np.column_stack([x_test, y_test])
    
    clf = [tree.DecisionTreeClassifier(),AdaBoostClassifier(),tree.DecisionTreeClassifier(max_depth=4)]    #基础模型
    #clf = [tree.DecisionTreeClassifier()]
    clf_self = Bagging(n_estimators = 100, estimator = clf,rate =1.0)
    num = len(clf_self.estimator) 
    if(len(clf_self.estimator) == 1):
        print "bagging有一个基础模型"
    elif(len(clf_self.estimator)>1):
        print "bagging有{}个基础模型".format(num)
    else:
        print "请输出基础模型"
    recall_self,precision_self = clf_self.MutModel_clf(train,test)
    print "recall:",'\n',recall_self
    print "precision",'\n',precision_self 
    
    #sklearn中 BaggingClassifier
    clf_sklearn = BaggingClassifier(base_estimator = tree.DecisionTreeClassifier(),n_estimators = 100)
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