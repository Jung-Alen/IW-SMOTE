# Aimin Zhang
# Time:2022/4/2


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import smote_variants as sv
import scipy.io as io
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import sklearn.metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import  precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import math
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib import gridspec
from   sklearn.model_selection  import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from time import *
from sklearn.neighbors import KNeighborsClassifier



class Compared_smote():
    def __init__(self, balance=1):
        self.balance = balance;


    def smote(self, data, target):
        self.data = data
        self.target = target
        df = pd.DataFrame(self.data)
        df[len(df.columns)] = self.target
        oversampler = sv.SMOTE(proportion = self.balance)
        X_samp, y_samp = oversampler.sample(np.array(df.iloc[:, 0:len(df.columns) - 1]), np.array(df[len(df.columns) - 1]))
        return X_samp, y_samp

    def b1_smote(self, data, target):
        self.data = data
        self.target = target
        df = pd.DataFrame(self.data)
        df[len(df.columns)] = self.target
        oversampler = sv.Borderline_SMOTE1(proportion=self.balance)
        X_samp, y_samp = oversampler.sample(np.array(df.iloc[:, 0:len(df.columns) - 1]),
                                            np.array(df[len(df.columns) - 1]))
        return X_samp, y_samp

    def b2_smote(self, data, target):
        self.data = data
        self.target = target
        df = pd.DataFrame(self.data)
        df[len(df.columns)] = self.target
        oversampler = sv.Borderline_SMOTE2(proportion=self.balance)
        X_samp, y_samp = oversampler.sample(np.array(df.iloc[:, 0:len(df.columns) - 1]),
                                            np.array(df[len(df.columns) - 1]))
        return X_samp, y_samp

    def sl_smote(self, data, target):
        self.data = data
        self.target = target
        df = pd.DataFrame(self.data)
        df[len(df.columns)] = self.target
        oversampler = sv.Safe_Level_SMOTE(proportion=self.balance)
        X_samp, y_samp = oversampler.sample(np.array(df.iloc[:, 0:len(df.columns) - 1]),
                                            np.array(df[len(df.columns) - 1]))
        return X_samp, y_samp

    def sn_smote(self, data, target):
        self.data = data
        self.target = target
        df = pd.DataFrame(self.data)
        df[len(df.columns)] = self.target
        oversampler = sv.SN_SMOTE(proportion=self.balance)
        X_samp, y_samp = oversampler.sample(np.array(df.iloc[:, 0:len(df.columns) - 1]),
                                            np.array(df[len(df.columns) - 1]))
        return X_samp, y_samp

    def mwmote(self, data, target):
        self.data = data
        self.target = target
        df = pd.DataFrame(self.data)
        df[len(df.columns)] = self.target
        oversampler = sv.MWMOTE(proportion=1.0, k1=5, k2=3, k3=int(len(df[df[len(df.columns)-1]==1])/2), M=3, cf_th=5.0, cmax=2.0, proportion=self.balance)
        X_samp, y_samp = oversampler.sample(np.array(df.iloc[:, 0:len(df.columns) - 1]),
                                            np.array(df[len(df.columns) - 1]))
        return X_samp, y_samp

    def adasyn(self, data, target):
        self.data = data
        self.target = target
        df = pd.DataFrame(self.data)
        df[len(df.columns)] = self.target
        oversampler = sv.ADASYN(n_neighbors=5, d_th=0.75, beta=1.0, proportion=self.balance)
        X_samp, y_samp = oversampler.sample(np.array(df.iloc[:, 0:len(df.columns) - 1]),
                                            np.array(df[len(df.columns) - 1]))
        return X_samp, y_samp

    def smote_enn(self, data, target):
        self.data = data
        self.target = target
        df = pd.DataFrame(self.data)
        df[len(df.columns)] = self.target
        oversampler = sv.SMOTE_ENN(proportion=self.balance)
        X_samp, y_samp = oversampler.sample(np.array(df.iloc[:, 0:len(df.columns) - 1]),
                                            np.array(df[len(df.columns) - 1]))
        return X_samp, y_samp

    def smote_tl(self, data, target):
        self.data = data
        self.target = target
        df = pd.DataFrame(self.data)
        df[len(df.columns)] = self.target
        oversampler = sv.SMOTE_TomekLinks(proportion=self.balance)
        X_samp, y_samp = oversampler.sample(np.array(df.iloc[:, 0:len(df.columns) - 1]),
                                            np.array(df[len(df.columns) - 1]))
        return X_samp, y_samp

    def smote_IPF(self, data, target):
        self.data = data
        self.target = target
        df = pd.DataFrame(self.data)
        df[len(df.columns)] = self.target
        oversampler = sv.SMOTE_IPF(n_neighbors=5, n_folds=9, k=3, p=0.01, voting='consensus', proportion=self.balance)
        X_samp, y_samp = oversampler.sample(np.array(df.iloc[:, 0:len(df.columns) - 1]),
                                            np.array(df[len(df.columns) - 1]))
        return X_samp, y_samp

    


