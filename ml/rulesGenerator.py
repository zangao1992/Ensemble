## Rules
import decisionTree as dt
import numpy as np
from sklearn.externals import joblib
import pandas as pd

#决策树提取 根节点到叶子节点的判断路径
class Rules:

    def __init__(self, clf, important_features):

        self.tree = clf.tree_

        self.lc = self.tree.children_left

        self.rc = self.tree.children_right

        self.tree_threshold =  self.tree.threshold

        self.feature = self.tree.feature

        self.rules = None

    def get_feature_by_importance(self, number, result_features):

        index = self.feature[number]

        return result_features.variable[index]

    def get_leaf_location(self):

        return filter(lambda i: self.lc[i] == self.rc[i],
                        range(self.tree.node_count))

    def find_parent(self, number):

        parent_index = np.where(self.lc == number)

        if not parent_index[0]:

            parent_index = np.where(self.rc == number)

        return parent_index[0]

    def find_path_to_root(self, number):

        path = list()

        path.append(number)

        parent = self.find_parent(number)

        while parent:

            path.append(parent[0])

            parent = self.find_parent(parent)

        path.append(0)

        return path

    def find_all_path_from_leaves(self):

        self.all_path = list()

        leaves = self.get_leaf_location()

        for i in leaves:

            self.all_path.append(self.find_path_to_root(i))

    def find_sign(self):

        self.all_sign = list()

        for array in self.all_path:

            sign = list()

            i = len(array) - 1

            while i >= 1:

                index = array[i - 1]

                if int(index) == int(self.lc[array[i]]):

                    sign.append(" <= ")

                elif int(index) == int(self.rc[array[i]]):

                    sign.append(" > ")

                i = i - 1

            self.all_sign.append(sign)

    def predict(self):

        self.predict = list()

        for array in self.all_path:

            predict_value = self.tree.value[array[0]]

            max_pre = max(predict_value[0])

            if max_pre == predict_value[0][0]:

                self.predict.append(0)

            elif max_pre == predict_value[0][1]:

                self.predict.append(1)

    def rule_generate(self, important_features):

        self.find_all_path_from_leaves()

        self.find_sign()

        self.predict()

        # i = 0
        rules = list()
        # self.support_value = list()
        # self.precision_value = list()
        # self.result = list()

        for i, array in enumerate(self.all_path):
            j = len(self.all_sign[i]) - 1

            conds = []

            for index in array[1:]:

                conds.append(self.get_feature_by_importance(index, important_features) +
                             self.all_sign[i][j] +
                             str(self.tree.threshold[index]))

                j = j - 1

            rule = (self.predict[i],
                    sum(self.tree.value[array[0]][0]),
                    self.tree.value[array[0]][0][self.predict[i]] / sum(self.tree.value[array[0]][0]),
                    ' && '.join(conds[::-1]))

            rules.append(rule)

        self.rules = sorted(rules, key=lambda r: r[1], reverse=True)


    def print_all(self):

        for rule in self.rules:

            print '\t '.join(map(str, rule))

