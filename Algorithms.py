'''
# @author：Aimin Zhang，Hualong Yu
# Time:2022/4/2
# Institution：Jiangsu University of Science and Technology
'''

import numpy as np
import pandas as pd
import smote_variants as sv
import random
import CSELM as ce
from sklearn.tree import DecisionTreeClassifier
import WKSMOTE as ws



# the classfication and regression tree
def CART(X, y, XX):
    model = DecisionTreeClassifier()
    model.fit(X, y)
    predicted = model.predict(XX)
    return predicted


class Compared_smote():
    def __init__(self, data, target, balance=1):
        self.balance = balance;  # Sampling rate
        self.data = data # The training set
        self.target = target # The labels of training set

    """
            :param lamda: lamda*imbalance ratio = the number of cart
            :param thres: The threshold of filtering noise
            :param k_neighbor: k nearest neighbor
            :param divide_times: The ratio of under sampling minority samples
            :param gen_times: The ratio of generated minority class to majority class samples
            :return: The synthetic samples, attributes and labels
    """
    def IW_SMOTE(self, lamda=100, thres=0.5, divide_times=2, gen_times=1, k_neighbor=5):
        # x_x:Temporary variable, save the training set
        x_x = pd.DataFrame(self.data)
        x_x[len(x_x.columns)] = self.target
        m, n = len(x_x), len(x_x.columns) #m: the number of trainning set; n: the number of columns of the trainning set
        z = x_x[x_x[n - 1] == 1]  # acquire the minority set
        p = x_x[x_x[n - 1] == -1]  # acquire the majority set
        m1, n1 = len(z), len(z.columns) # m1: the number of minority samples; n1: the number of columns of the minority samples
        m2, n2 = len(p), len(p.columns) # m2: the number of majority samples; n2: the number of columns of the majority samples
        IR = m2 / m1  # imbalance ratio
        predict_min_labelset = pd.DataFrame(columns = range(int(IR * lamda))) # the predicted labels of the minority samples
        predict_maj_labelset = pd.DataFrame(columns = range(int(IR * lamda))) # the predicted labels of the majority samples
        # train under-bagging CART
        for i_1 in range(int(IR * lamda)):
            train_subset = z.sample(int(m1 / divide_times)) # train_subset: the subset of training set
            train_subset = train_subset.append(p.sample(int(m1 / divide_times), replace=True))
            predict_maj_labelset[i_1] = CART(np.array(train_subset.iloc[:, 0:n1 - 1]), np.array(train_subset[n1 - 1]),
                                             np.array(p.iloc[:, 0:n2 - 1]))
            predict_min_labelset[i_1] = CART(np.array(train_subset.iloc[:, 0:n1 - 1]), np.array(train_subset[n1 - 1]),
                                             np.array(z.iloc[:, 0:n1 - 1]))
        # filterring noise
        err_rate_min = []  # record the error rate of the reserved minority instance
        reserve_min = []  # record the reserved minority instances
        num_reserve_min = 0  # the number of minority samples after denoising
        z1 = np.array(z)
        predict_min_labelset = np.array(predict_min_labelset)
        for i_2 in range(m1):
            num_right = 0  # record the number of instance which is predicted accurately
            for j in range(int(IR * lamda)):
                if predict_min_labelset[i_2][j] == z1[i_2][n1 - 1]:
                    num_right = num_right + 1
            if ((int(IR * lamda) - num_right) / int(IR * lamda) < thres):
                num_reserve_min += 1
                reserve_min.append(z1[i_2])
                if (int(IR * lamda) - num_right) / int(IR * lamda) < 1 / int(IR * lamda):
                    err_rate_min.append(1 / int(IR * lamda))
                else:
                    err_rate_min.append((int(IR * lamda) - num_right) / int(IR * lamda))
        reserve_min = pd.DataFrame(reserve_min)
        err_rate_min = pd.DataFrame(err_rate_min)
        err_rate_maj = []  # record the error rate of the reserved minority instance
        reserve_maj = []  # record the reserved minority instances
        num_reserve_maj = 0  # the number of majority samples after denoising
        p1 = np.array(p)
        predict_maj_labelset = np.array(predict_maj_labelset)
        for i_3 in range(m2):
            num_right = 0  # record the number of instance which is predicted accurately
            for j in range(int(IR * lamda)):
                if predict_maj_labelset[i_3][j] == p1[i_3][n2 - 1]:
                    num_right = num_right + 1
            if ((int(IR * lamda) - num_right) / int(IR * lamda) < thres):
                num_reserve_maj += 1
                reserve_maj.append(p1[i_3])
                if (int(IR * lamda) - num_right) / int(IR * lamda) < 1 / int(IR * lamda):
                    err_rate_maj.append(1 / int(IR * lamda))
                else:
                    err_rate_maj.append((int(IR * lamda) - num_right) / int(IR * lamda))
        reserve_maj = pd.DataFrame(reserve_maj)

        # generate the synthetic minority instances
        weight = err_rate_min[0] / sum(err_rate_min[0])  # Record the importance of each sample
        num_need_generate = gen_times * num_reserve_maj - num_reserve_min  # The number of minority samples that need to be synthesized
        if num_need_generate == num_reserve_maj:
            return np.array(reserve_maj.iloc[:, 0:len(reserve_maj.columns) - 1]), np.array(
                reserve_maj[len(reserve_maj.columns) - 1])
        else:
            num_generate = 0 # the number of be generated
            new_set = pd.DataFrame(columns=range(n1))
            for i_4 in range(num_reserve_min):
                reserve_min_1 = reserve_min
                nums = pd.DataFrame(weight * (gen_times * num_reserve_maj - num_reserve_min)).iloc[i_4, 0]
                reserve_min_1 = np.array(reserve_min_1)
                dis = [0] * num_reserve_min # distance matrix of per sample
                for m in range(num_reserve_min):
                    dis[m] = np.linalg.norm(reserve_min_1[i_4] - reserve_min_1[m])
                b = sorted(enumerate(dis), key=lambda xxx: xxx[1]) #Sorted dis
                b = b[1:k_neighbor + 1] # choice knn
                for j in range(int(nums)):
                    num_generate = num_generate + 1
                    s_b = random.choice(b)
                    select_ins = reserve_min.iloc[s_b[0], :]
                    new_ins = (reserve_min.iloc[i_4, :] - pd.DataFrame(select_ins).T) * random.random() + pd.DataFrame(
                        select_ins).T  # generate funtion
                    new_set = new_set.append(pd.DataFrame(new_ins))  # add the new instance into a temporary set
            new_z = reserve_min.append(new_set) # The minority synthetic samples
            new_original_data = reserve_maj.append(new_z) # The synthetic samples
            new_original_data.index = range(len(new_original_data))
            # Returns an oversampled dataset
            return np.array(new_original_data.iloc[:, 0:len(new_original_data.columns) - 1]), np.array(new_original_data[len(new_original_data.columns) - 1])


    def SMOTE(self):
        """
                    :return: The synthetic samples
                    :X_samp: The attribute of the synthetic samples
                    :y_samp: The labels of the synthetic samples
        """
        df = pd.DataFrame(self.data)
        df[len(df.columns)] = self.target
        oversampler = sv.SMOTE(proportion=self.balance)
        X_samp, y_samp = oversampler.sample(np.array(df.iloc[:, 0:len(df.columns) - 1]), np.array(df[len(df.columns) - 1]))
        return X_samp, y_samp


    def B1_SMOTE(self):
        """
                            :return: The synthetic samples
                            :X_samp: The attribute of the synthetic samples
                            :y_samp: The labels of the synthetic samples
        """
        df = pd.DataFrame(self.data)
        df[len(df.columns)] = self.target
        oversampler = sv.Borderline_SMOTE1(proportion=self.balance)
        X_samp, y_samp = oversampler.sample(np.array(df.iloc[:, 0:len(df.columns) - 1]), np.array(df[len(df.columns) - 1]))
        return X_samp, y_samp


    def B2_SMOTE(self):
        """
                            :return: The synthetic samples
                            :X_samp: The attribute of the synthetic samples
                            :y_samp: The labels of the synthetic samples
        """
        df = pd.DataFrame(self.data)
        df[len(df.columns)] = self.target
        oversampler = sv.Borderline_SMOTE2(proportion=self.balance)
        X_samp, y_samp = oversampler.sample(np.array(df.iloc[:, 0:len(df.columns) - 1]), np.array(df[len(df.columns) - 1]))
        return X_samp, y_samp


    def SL_SMOTE(self):
        """
                            :return: The synthetic samples
                            :X_samp: The attribute of the synthetic samples
                            :y_samp: The labels of the synthetic samples
        """
        df = pd.DataFrame(self.data)
        df[len(df.columns)] = self.target
        oversampler = sv.Safe_Level_SMOTE(proportion=self.balance)
        X_samp, y_samp = oversampler.sample(np.array(df.iloc[:, 0:len(df.columns) - 1]), np.array(df[len(df.columns) - 1]))
        return X_samp, y_samp


    def SN_SMOTE(self):
        """
                            :return: The synthetic samples
                            :X_samp: The attribute of the synthetic samples
                            :y_samp: The labels of the synthetic samples
        """
        df = pd.DataFrame(self.data)
        df[len(df.columns)] = self.target
        oversampler = sv.SN_SMOTE(proportion=self.balance)
        X_samp, y_samp = oversampler.sample(np.array(df.iloc[:, 0:len(df.columns) - 1]), np.array(df[len(df.columns) - 1]))
        return X_samp, y_samp


    def MWMOTE(self):
        """
                            :return: The synthetic samples
                            :X_samp: The attribute of the synthetic samples
                            :y_samp: The labels of the synthetic samples
        """
        df = pd.DataFrame(self.data)
        df[len(df.columns)] = self.target
        oversampler = sv.MWMOTE(proportion=self.balance, k1=5, k2=3, k3=int(len(df[df[len(df.columns) - 1] == 1]) / 2), M=3, cf_th=5.0, cmax=2.0)
        X_samp, y_samp = oversampler.sample(np.array(df.iloc[:, 0:len(df.columns) - 1]), np.array(df[len(df.columns) - 1]))
        return X_samp, y_samp


    def ADASYN(self):
        """
                            :return: The synthetic samples
                            :X_samp: The attribute of the synthetic samples
                            :y_samp: The labels of the synthetic samples
        """
        df = pd.DataFrame(self.data)
        df[len(df.columns)] = self.target
        oversampler = sv.ADASYN(n_neighbors=5, d_th=0.75, beta=1.0, proportion=self.balance)
        X_samp, y_samp = oversampler.sample(np.array(df.iloc[:, 0:len(df.columns) - 1]), np.array(df[len(df.columns) - 1]))
        return X_samp, y_samp


    def SMOTE_ENN(self):
        """

                            :return: The synthetic samples
                            :X_samp: The attribute of the synthetic samples
                            :y_samp: The labels of the synthetic samples
        """
        df = pd.DataFrame(self.data)
        df[len(df.columns)] = self.target
        oversampler = sv.SMOTE_ENN(proportion=self.balance)
        X_samp, y_samp = oversampler.sample(np.array(df.iloc[:, 0:len(df.columns) - 1]), np.array(df[len(df.columns) - 1]))
        return X_samp, y_samp


    def SMOTE_TL(self):
        """
                            :return: The synthetic samples
                            :X_samp: The attribute of the synthetic samples
                            :y_samp: The labels of the synthetic samples
        """
        df = pd.DataFrame(self.data)
        df[len(df.columns)] = self.target
        oversampler = sv.SMOTE_TomekLinks(proportion=self.balance)
        X_samp, y_samp = oversampler.sample(np.array(df.iloc[:, 0:len(df.columns) - 1]), np.array(df[len(df.columns) - 1]))
        return X_samp, y_samp


    def SMOTE_IPF(self):
        """
                            :return: The synthetic samples
                            :X_samp: The attribute of the synthetic samples
                            :y_samp: The labels of the synthetic samples
        """
        df = pd.DataFrame(self.data)
        df[len(df.columns)] = self.target
        oversampler = sv.SMOTE_IPF(n_neighbors=5, n_folds=9, k=3, p=0.01, voting='consensus', proportion=self.balance)
        X_samp, y_samp = oversampler.sample(np.array(df.iloc[:, 0:len(df.columns) - 1]), np.array(df[len(df.columns) - 1]))
        return X_samp, y_samp

    def SMOTE_CSELM(self, test):
        """
                            :test: The testing set without labels
                            :return: The predict labels of the testing set
        """
        df = pd.DataFrame(self.data)
        df[len(df.columns)] = self.target
        x_x = df
        m, n = len(x_x), len(x_x.columns)
        z = x_x[x_x[n - 1] == 1]  # acquire the minority set
        p = x_x[x_x[n - 1] == -1]  # acquire the majority set
        oversampler = sv.SMOTE(proportion=self.balance)
        X_samp, y_samp = oversampler.sample(np.array(df.iloc[:, 0:len(df.columns) - 1]),
                                            np.array(df[len(df.columns) - 1]))
        syn = pd.DataFrame(X_samp)
        syn[len(syn.columns)] = y_samp
        syn = pd.concat([z, p, syn])
        syn = syn.drop_duplicates(keep=False)
        cse = ce.CSELM(z.iloc[:, :n - 1], z.iloc[:, :n - 1], syn.iloc[:, :n - 1], 100)
        cse.classifisor_train(z[n - 1].append(p[n - 1]).append(syn[n - 1])) #train classifier on balance dataset
        return cse.classifisor_test(test)


    def WKSMOTE(self, test):
        """
                            :test: The testing set without labels
                            :return: The predict labels of the testing set
        """
        wk_smote = ws.WKSMOTE(data = self.data, target = self.target, test = test, balance=self.balance)
        wk_smote.train()
        return wk_smote.predict()




