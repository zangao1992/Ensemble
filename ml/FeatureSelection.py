#coding:utf-8

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
def boolean2int(df):
	return df.replace("false", 0).replace("true", 1).replace("False", 0).replace("True", 1)

def count_drop_nan(df,feature,threshold = 0.5,drop = False):
	#输入某一维度属性值，统计该属性缺失值占总数比例 threshold为阈值，当比例大于该值时并且dorp = True 删去该特征
	percent = df[feature].isnull().sum(axis = 1)/df[feature].shape[0]
	print "属性的缺失值占总数的比例：", percent
	if drop ==True and percent >= threshold:
		print "删除特征：",feature
		df.drop(feature,axis=1,inplace=True)

#填充缺失值
def fillna(df):
	return df.fillna(0).replace("NULL", 0).replace("null", 0)

#填充数值类型的缺失值
def fillna_num(df,feature):
	print "默认填充方法：data.median()"
	return df[feature].fillna(function = df.median())	

#连续值进行离散化
def discretize(df,feature, segments, prefix):
	min = 0
	idx = 0
	columns_names = df.columns.tolist()
	for seg in segments:
		df.insert(columns_names.index(feature) + idx, prefix + '_' + str(seg), (df[feature]> min) & (df[feature]<=seg))
		min = seg
		idx += 1

def feature_kbest(df,label):
	#卡方检验特征选择方法:
	from sklearn.feature_selection import SelectKBest
	from sklearn.feature_selection import chi2
	f_sel = SelectKBest(chi2, k=5).fit(df, label)
	print "卡方检验特征名和其评分为："
	print pd.DataFrame(df).columns
	print f_sel.scores_
	dic=dict(zip(pd.DataFrame(df).columns,f_sel.scores_))
	dic = sorted(dic.items(),key=lambda item:item[1],reverse=True)
	print dic
	return dic

def feature_variance(df):
	#方差选择法，返回评分：
	from sklearn.feature_selection import VarianceThreshold
	var = VarianceThreshold().fit(df)
	print "方差选择法、特征名和其方差值为："
	print pd.DataFrame(df).columns
	print var.variances_
	dic=dict(zip(pd.DataFrame(df).columns,var.variances_))
	dic = sorted(dic.items(),key=lambda item:item[1],reverse=True)
	print dic
	return dic

def feature_mine(df,label):
	#特征选择，互信息法
	from sklearn.feature_selection import SelectKBest
 	from minepy import MINE
 	#由于MINE的设计不是函数式的，定义mic方法将其为函数式的，返回一个二元组，二元组的第2项设置成固定的P值0.5
	def mic(x, y):
	    m = MINE()
	    m.compute_score(x, y)
	    return (m.mic(), 0.5)
	#选择K个最好的特征，返回特征选择后的数据
	f_sel = SelectKBest(lambda X, Y: np.array(map(lambda x:mic(x, Y), X.T)).T, k=2).fit(df, label)
	print "互信息法特征名和其评分为："
	dic=dict(zip(pd.DataFrame(df).columns,f_sel.scores_[0]))
	dic = sorted(dic.items(),key=lambda item:item[1],reverse=True)
	print dic
	return dic

def feature_regular(df,label):
	#基于惩罚项的特征选择法
	#结合L1和L2惩罚项的逻辑回归作为基模型的特征选择
	class LR(LogisticRegression):
	    def __init__(self, threshold=0.01, dual=False, tol=1e-4, C=1.0,
	                 fit_intercept=True, intercept_scaling=1, class_weight=None,
	                 random_state=None, solver='liblinear', max_iter=100,
	                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):

	        #权值相近的阈值
	        self.threshold = threshold
	        LogisticRegression.__init__(self, penalty='l1', dual=dual, tol=tol, C=C,
	                 fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
	                 random_state=random_state, solver=solver, max_iter=max_iter,
	                 multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
	        #使用同样的参数创建L2逻辑回归
	        self.l2 = LogisticRegression(penalty='l2', dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight = class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

	    def fit(self, X, y, sample_weight=None):
	        #训练L1逻辑回归
	        super(LR, self).fit(X, y, sample_weight=sample_weight)
	        self.coef_old_ = self.coef_.copy()
	        #训练L2逻辑回归
	        self.l2.fit(X, y, sample_weight=sample_weight)

	        cntOfRow, cntOfCol = self.coef_.shape
	        #权值系数矩阵的行数对应目标值的种类数目
	        for i in range(cntOfRow):
	            for j in range(cntOfCol):
	                coef = self.coef_[i][j]
	                #L1逻辑回归的权值系数不为0
	                if coef != 0:
	                    idx = [j]
	                    #对应在L2逻辑回归中的权值系数
	                    coef1 = self.l2.coef_[i][j]
	                    for k in range(cntOfCol):
	                        coef2 = self.l2.coef_[i][k]
	                        #在L2逻辑回归中，权值系数之差小于设定的阈值，且在L1中对应的权值为0
	                        if abs(coef1-coef2) < self.threshold and j != k and self.coef_[i][k] == 0:
	                            idx.append(k)
	                    #计算这一类特征的权值系数均值
	                    mean = coef / len(idx)
	                    self.coef_[i][idx] = mean
	        return self
	from sklearn.feature_selection import SelectFromModel	
	f_sel = SelectFromModel(LR(threshold=0.5, C=0.1)).fit_transform(df,label)
	print "特征名和其是否被选择："
	print pd.DataFrame(df).columns
	print f_sel.get_support()
	dic=dict(zip(pd.DataFrame(df).columns,f_sel.get_support()))
	print dic
	return dic

def feature_rfe(df,label,feature_num,estimator=LogisticRegression()):
	#Wrapper：递归特征消除法，返回特征选择后的数据 
	#参数estimator为基模型
	#参数feature_num为选择的特征个数
	from sklearn.feature_selection import RFE
	print "基础分类器为:",estimator
	selector = RFE(estimator,feature_num).fit(df,label)
	dic=dict(zip(pd.DataFrame(df).columns,selector.support_ ))
	dic = sorted(dic.items(),key=lambda item:item[1],reverse=True)
	print "递归特征消除法:特征名和其是否被模型选择(1为最好)："
	print dic
	return dic

def feature_rf(df,label):
	from sklearn.ensemble import RandomForestClassifier
	print "正在训练随机森林："
	clf = RandomForestClassifier(n_estimators=100)
	print clf
	clf.fit(df,label)
	dic=dict(zip(pd.DataFrame(df).columns,clf.feature_importances_))
	dic = sorted(dic.items(),key=lambda item:item[1],reverse=True)
	print "随机森林MDG属性重要度:特征名和其重要度："
	print dic
	return dic

def PCA(df, dim):
	#PCA适合用在样本数量大，并且样本为指数组分布的数据。
    print "适用PCA前请统一量纲，PCA适用于指数组分布的数据"
    print "PCA降维，主成分数目为：",dim
    from sklearn.decomposition import PCA
    newdf = PCA(n_components=dim).fit_transform(df)	
    return newdf

'''
def LDA(df,label,dim):
	print "LDA降维，降维后的维数为：",dim
	from sklearn.lda import LDA
	newdf = LDA(n_components=dim).fit_transform(iris.data, iris.target)
	return newdf
'''

def sub_kmeans(df,number):
	#number 为欠采样后的数量，df为原数据中多数类
	from sklearn.cluster import KMeans
	number = int(number)
	print "聚类欠采样前样本的规模：",df.shape
	kmeans = KMeans(n_clusters=number).fit(df)
	newdf=kmeans.cluster_centers_
	print "聚类欠采样后样本的规模：",newdf.shape
	return newdf

class GBDTTransformer(object):
	#GBDT 提取特征，需要先训练好GBDT，然后将模型赋给transform()函数
    def __init__(self, estimator):
        self.estimator = estimator

    def transform_by_tree(self, estimator, X):
        n_nodes = estimator.tree_.node_count
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right

        # 找到叶子节点
        leaves = filter(lambda i: children_left[i] == children_right[i], 
                                range(n_nodes))

        # 样本最终落到树的哪个叶子节点
        leave_id = estimator.apply(X)

        # 根据叶子节点序号得到新特征
        return (leave_id.reshape((-1,1)).repeat(len(leaves), 1) - 
                            np.array(leaves) == 0).astype('int')

    def transform(self, X):
        estimator_feats = map(
                lambda estimator: self.transform_by_tree(estimator[0], X), 
                self.estimator.estimators_)
    
        return np.hstack(estimator_feats)

if __name__ == "__main__":
	print "main"
	data = {'height':np.random.randint(40, 50, size = 10), 'weight':np.random.randint(150, 180,size=10)}
	df = pd.DataFrame(data)
	discretize(df, "weight", [159, 169, 180], "X")
	print df